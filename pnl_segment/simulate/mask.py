import pathlib

import nibabel as nib
import numpy as np

import pnl_segment
from pnl_segment.point_cloud.ref_space import RefSpace


def check_ref(fnc):
    def wrapped(self, other):
        if hasattr(other, 'ref_space'):
            if self.ref_space != other.ref_space:
                raise AttributeError('incompatible ref space')

        fnc_array = getattr(self.x, fnc.__name__)

        if isinstance(other, Mask):
            other = other.x

        return Mask(fnc_array(other), ref_space=self.ref_space)

    return wrapped


class Mask:
    """ array

    >>> m1 = Mask(np.eye(3))
    >>> data = np.arange(18).reshape((3, 3, 2))
    >>> data[:, :, 0]
    array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])
    >>> m1.apply(data)
    array([[ 0,  1],
           [ 8,  9],
           [16, 17]])
    >>> data_inserted = m1.insert(data, np.arange(6) * -1)
    >>> data_inserted[:, :, 0]
    array([[ 0,  2,  4],
           [ 6, -2, 10],
           [12, 14, -4]])
    >>> m2 = m1 + m1 * 10 - (m1 / 10)
    >>> m2.x
    array([[10.9,  0. ,  0. ],
           [ 0. , 10.9,  0. ],
           [ 0. ,  0. , 10.9]])
    >>> folder = pathlib.Path(pnl_segment.__file__).parent
    >>> f_nii = folder / 'test' / 'data' / 'af.right.mask.nii.gz'
    >>> mask = Mask.from_nii(f_nii)
    >>> mask.ref_space.affine
    array([[ -2.        ,   0.        ,   0.        , 120.5       ],
           [  0.        ,  -2.        ,   0.        , 146.5       ],
           [  0.        ,   0.        ,   2.        , -88.90000153],
           [  0.        ,   0.        ,   0.        ,   1.        ]])
    """

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return np.sum((self.x > 0).flatten())

    def __init__(self, x, ref_space=None):
        self.x = x
        self.ref_space = ref_space

    @check_ref
    def __add__(self, other):
        pass

    @check_ref
    def __mul__(self, other):
        pass

    @check_ref
    def __truediv__(self, other):
        pass

    @check_ref
    def __sub__(self, other):
        pass

    @check_ref
    def __radd__(self, other):
        pass

    @check_ref
    def __rsub__(self, other):
        pass

    @check_ref
    def __rmul__(self, other):
        pass

    @staticmethod
    def from_nii(f_nii):
        img = nib.load(str(f_nii))
        ref = RefSpace(affine=img.affine, shape=img.shape)
        return Mask(img.get_data(), ref_space=ref)

    def __iter__(self):
        for ijk in np.vstack(np.where(self.x > 0)).T:
            yield tuple(ijk)

    def iter_multidim(self, n):
        for ijk in self:
            for _n in range(n):
                yield (*ijk, _n)

    def insert(self, data_img, x, add=False):
        """ inserts x into data_img where mask is

        Args:
            data_img (np.array):
            x (np.array):
            add (bool): toggles whether data is replaced or added

        Returns:
            data_img
        """
        # iterate through idx
        if len(data_img.shape) > len(self.shape):
            if len(data_img.shape) > len(self.shape) + 1:
                raise AttributeError('data img has more than 1 dim than mask')
            d = data_img.shape[-1]
            ijk_iter = self.iter_multidim(d)
        else:
            d = 1
            ijk_iter = iter(self)

        if len(x) != len(self) * d:
            raise AttributeError('size mismatch, data len does not match mask')

        # insert given data
        for _x, ijk in zip(x, ijk_iter):
            if add:
                data_img[ijk] += _x
            else:
                data_img[ijk] = _x

        return data_img

    def apply(self, data):
        # validate mask shape
        if not np.array_equal(self.shape, data.shape[:len(self.shape)]):
            raise AttributeError('shape mismatch')

        if len(data.shape) > len(self.shape) + 1:
            raise AttributeError('data shape more than 1 dim larger than mask')

        return np.vstack(data[np.s_[ijk]] for ijk in self)

    def apply_from_nii(self, f_nii):
        img = nib.load(str(f_nii))
        if self.ref_space.affine is not None and \
                not np.array_equal(img.affine, self.ref_space.affine):
            raise AttributeError('affine mismatch')

        return self.apply(img.get_data())
