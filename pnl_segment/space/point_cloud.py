import nibabel as nib
import numpy as np

from .ref_space import RefSpace, get_ref


class PointCloud:
    """ super class for PointCloudIJK and PointCloudXYZ

    >>> x =np.vstack((np.eye(2), np.eye(2)))
    >>> print(x)
    [[1. 0.]
     [0. 1.]
     [1. 0.]
     [0. 1.]]
    >>> pc = PointCloud(x)
    >>> print(pc)
    PointCloud with 4 pts
    >>> len(pc)
    4
    >>> pc.dim
    2
    >>> pc.center
    array([0.5, 0.5])
    >>> pc.discard_doubles()
    >>> print(pc)
    PointCloud with 2 pts
    """

    def __init__(self, x, ref):
        if len(x.shape) == 1:
            raise AttributeError('must init with 2d array')

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        self.x = x
        self.ref = get_ref(ref)

    def swap_ref(self, ref):
        """ swaps orientation to a new ref
        """
        x = self.ref.to_rasmm(self.x)

        ref_to = get_ref(ref)
        x = ref_to.from_rasmm(x)

        return self.__class__(x, ref=ref)

    def __len__(self):
        return self.x.shape[0]

    @property
    def dim(self):
        return self.x.shape[1]

    @property
    def center(self):
        return np.mean(self.x, axis=0)

    def __str__(self):
        return f'{type(self).__name__} with {len(self)} pts'

    def discard_doubles(self):
        self.x = np.unique(self.x, axis=0)

    def __eq__(self, other):
        # note: resorting pts spoils equality
        return np.array_equal(self.x, other.x)

    def __add__(self, other, check_ref=False):
        """ adds points to the point cloud
        """
        if type(self) != type(other):
            TypeError(f'type mismatch: {type(self)}, {type(other)}')

        if not isinstance(other, PointCloud) and other == 0:
            # allows sum()
            return self.__class__(self.x, ref=self.ref)

        # ensure affines are equivilent
        if check_ref and self.ref != other.ref:
            raise AttributeError('cant merge different spaces')

        return self.__class__(x=np.vstack((self.x, other.x)), ref=self.ref)

    __radd__ = __add__


class PointCloudIJK(PointCloud):
    """ stores an array of voxels
    """

    def __iter__(self):
        for ijk in self.x:
            yield tuple(ijk)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ref.shape is None:
            raise AttributeError('PointCloudIJK requires ref with shape')

    def to_xyz(self, ref=None):
        """ maps into xyz space, returns PointCloudXYZ
        """
        if ref is None:
            # map into scanner space if no other space given
            ref = RefSpace(affine=np.eye(4))

        # map into scanner x_rasmm
        x_rasmm = self.ref.to_rasmm(self.x)

        # map into new space
        x = ref.from_rasmm(x_rasmm)

        return PointCloudXYZ(x, ref)

    @staticmethod
    def from_nii(f_nii, target_idx=None):
        """ read in point cloud from a mask
        """
        # get reference space
        ref = get_ref(f_nii)

        # read in volume
        img_mask = nib.load(str(f_nii))
        mask = img_mask.get_data()

        # constrain to only valid values
        if target_idx is None:
            # default: positive values
            mask = mask > 0
        else:
            mask = mask == target_idx

        # convert to ijk
        ijk = np.vstack(np.where(mask)).T

        # init point cloud
        return PointCloudIJK(ijk, ref)

    def to_nii(self, f_nii, **kwargs):
        """ writes mask to nifti file"""

        x = self.to_array(**kwargs)
        img = nib.Nifti1Image(x, self.ref.affine)
        img.to_filename(str(f_nii))

    def swap_ref(self, ref):
        """ returns pc_ijk whose ref is swapped
        """
        pc_ijk = super().swap_ref(ref)

        # round
        pc_ijk.x = np.rint(pc_ijk.x).astype(int)

        return pc_ijk

    def to_array(self, idx_err_flag=True, dtype=np.uint16):
        """ builds mask array, default 1 in voxels with pt present, 0 otherwise

        Args:
            idx_err_flag (bool): toggles if error on ijk out of self.ref.shape
            dtype (class): data type

        Returns:
            x (np.array): 1s where ijk, 0s elsewhere
        """

        # init array
        array = np.zeros(self.ref.shape, dtype=dtype)

        # throw error if any idx is outside (and idx_err_flag)
        for _x in self.x:
            try:
                # set mask to 1
                array[tuple(_x)] = 1
            except IndexError:
                if idx_err_flag:
                    # only throw error if idx_err_flag
                    IndexError(f'{_x} out of bounds in shape {self.ref.shape}')
        return array


class PointCloudXYZ(PointCloud):
    """ stores an array of points
    """

    def to_ijk(self, ref):
        """ maps into some ijk space (rounding to int), returns PointCloudIJK
        """

        ref = get_ref(ref)

        ijk = ref.from_rasmm(self.x, round_flag=True)

        return PointCloudIJK(ijk, ref)

    @staticmethod
    def from_nii(f_nii, **kwargs):
        """ read in nii mask
        """
        pc_ijk = PointCloudIJK.from_nii(f_nii, **kwargs)
        return pc_ijk.to_xyz()

    def to_nii(self, f_nii, ref, **kwargs):
        """ save nii mask
        """
        self.to_ijk(ref).to_nii(f_nii, **kwargs)

    @staticmethod
    def from_tract(f_trk):
        """ build from trk file
        """
        tract = nib.streamlines.load(str(f_trk))
        ref = get_ref(str(f_trk))

        if tract.streamlines:
            return sum(PointCloudXYZ(line, ref) for line in tract.streamlines)
        else:
            return PointCloudXYZ(np.zeros((0, 3)), ref)
