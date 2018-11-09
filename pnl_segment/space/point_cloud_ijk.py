import nibabel as nib
import numpy as np

from .point_cloud import PointCloud
from .point_cloud_xyz import PointCloudXYZ
from .ref_space import get_ref, RefSpace


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
