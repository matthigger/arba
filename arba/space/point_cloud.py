import nibabel as nib
import numpy as np

from .mask import Mask
from .ref_space import RefSpace, get_ref


class PointCloud(set):
    """ a set of points in space (e.g. voxels), stored as set

    terminology note:
    'mask_array' is a typical mask image, points are the ijk of non zero voxels
    'array' is a (n feat dim) array of points (either ijk of xyz)
    """

    @staticmethod
    def from_tract(f_trk):
        """ build from trk file
        """
        tract = nib.streamlines.load(str(f_trk))
        ref = get_ref(str(f_trk))

        # points assumed in xyz space
        ref.affine = np.eye(4)

        if tract.streamlines:
            pc_gen = (PointCloud.from_array(line)
                      for line in tract.streamlines)
            return PointCloud(set().union(*pc_gen), ref=ref)
        else:
            return PointCloud(np.zeros((0, 3)), ref)

    @staticmethod
    def from_mask(m):
        # convert to array of 3-tuple
        x = np.vstack(np.where(m)).T

        return PointCloud.from_array(x, ref=m.ref)

    @staticmethod
    def from_nii(f_nii):
        mask = Mask.from_nii(f_nii)
        return PointCloud.from_mask(mask)

    @staticmethod
    def from_array(x, **kwargs):
        gen = (tuple(_x) for _x in x)
        return PointCloud(gen, **kwargs)

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
        return f'{type(self).__name__} w/ {len(self)} pts'

    def swap_ref(self, ref=None, round=False):
        """ swaps orientation to a new ref
        """
        if ref is None:
            # defaults to rasmm with same shape
            ref = RefSpace(affine=np.eye(self.dim + 1), shape=self.ref.shape)

        x = self.ref.to_rasmm(self.to_array())

        ref_to = get_ref(ref)
        x = ref_to.from_rasmm(x)

        # round
        if round:
            x = np.rint(x).astype(int)

        return PointCloud.from_array(x, ref=ref)

    def to_array(self):
        return np.vstack(self)

    def to_mask(self, ref=None, shape=None, ignore_out=False):
        if ref is None:
            ref = self.ref
            pc = self
        else:
            ref = get_ref(ref)
            shape = ref.shape
            pc = self.swap_ref(ref)

        if shape is None:
            if self.ref is None or self.ref.shape is None:
                raise AttributeError('shape required')
            shape = self.ref.shape

        mask = np.zeros(shape)
        for _x in pc:
            _x = tuple(np.rint(_x).astype(int))
            try:
                mask[_x] = 1
            except IndexError:
                if not ignore_out:
                    raise IndexError

        return Mask(mask, ref=ref)

    # https://stackoverflow.com/questions/798442/what-is-the-correct-or-best-way-to-subclass-the-python-set-class-adding-a-new
    @classmethod
    def _wrap_methods(cls, names):
        def wrap_method_closure(name):
            def inner(self, other):
                result = getattr(super(cls, self), name)(other)
                return cls(result, ref=self.ref)

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
