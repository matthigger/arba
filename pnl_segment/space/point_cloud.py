import numpy as np

from .ref_space import RefSpace, get_ref


class PointCloud(set):
    """ a set of points in space (e.g. voxels), stored as set

    note:
    'mask_array' is a typical mask image
    'ijk_array' is a (n x dim) array of the points
    """

    @staticmethod
    def from_tract(f_trk):
        """ build from trk file
        """
        raise NotImplementedError
        # tract = nib.streamlines.load(str(f_trk))
        # ref = get_ref(str(f_trk))
        #
        # if tract.streamlines:
        #     return sum(PointCloudXYZ(line, ref) for line in tract.streamlines
        # else:
        #     return PointCloudXYZ(np.zeros((0, 3)), ref)

    @staticmethod
    def from_mask_array(x, **kwargs):
        # convert to array of ijk idx
        x = np.vstack(np.where(x)).T
        return PointCloud.from_ijk_array(x, **kwargs)

    @staticmethod
    def from_ijk_array(x, **kwargs):
        ijk_gen = (tuple(_x) for _x in x)
        return PointCloud(ijk_gen, **kwargs)

    def __init__(self, *args, ref=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.ref = get_ref(ref)

        if self:
            self.dim = len(next(iter(self)))
        elif self.ref is not None:
            self.dim = self.ref.affine.shape[0] - 1
        else:
            self.dim = None

    def __str__(self):
        return f'{type(self)} w/ {len(self)} pts'

    def swap_ref(self, ref=None, round=False):
        """ swaps orientation to a new ref
        """
        if ref is None:
            # defaults to rasmm with same shape
            ref = RefSpace(affine=np.eye(self.dim + 1), shape=self.ref.shape)

        x = self.ref.to_rasmm(self.to_ijk_array())

        ref_to = get_ref(ref)
        x = ref_to.from_rasmm(x)

        # round
        if round:
            x = np.rint(x).astype(int)

        return PointCloud.from_ijk_array(x, ref=ref)

    def to_mask_array(self, idx_err_flag=True):
        """ builds mask array, default 1 in voxels with pt present, 0 otherwise

        Args:
            idx_err_flag (bool): toggles if error on ijk out of self.ref.shape

        Returns:
            x (np.array): 1s where ijk, 0s elsewhere
        """

        # init array
        array = np.zeros(self.ref.shape, dtype=bool)

        # throw error if any idx is outside (and idx_err_flag)
        for x in self:
            try:
                # set mask to 1
                array[x] = 1
            except IndexError:
                if idx_err_flag:
                    # only throw error if idx_err_flag
                    IndexError(f'{x} out of bounds in shape {self.ref.shape}')
        return array

    def to_ijk_array(self):
        return np.vstack(self)

    # https://stackoverflow.com/questions/798442/what-is-the-correct-or-best-way-to-subclass-the-python-set-class-adding-a-new
    @classmethod
    def _wrap_methods(cls, names):
        def wrap_method_closure(name):
            def inner(self, other):
                result = getattr(super(cls, self), name)(other)
                if isinstance(result, set) and not hasattr(result, 'ref'):
                    assert self.ref == other.ref, 'unequal ref'
                    result = cls(result, ref=self.ref)
                return result

            inner.fn_name = name
            setattr(cls, name, inner)

        for name in names:
            wrap_method_closure(name)


PointCloud._wrap_methods(
    ['__ror__', 'difference_update', '__isub__', 'symmetric_difference',
     '__rsub__', '__and__', '__rand__', 'intersection', 'difference',
     '__iand__', 'union', '__ixor__', 'symmetric_difference_update',
     '__or__', 'copy', '__rxor__', 'intersection_update', '__xor__',
     '__ior__', '__sub__'])
