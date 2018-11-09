import tempfile

import nibabel as nib
import numpy as np

from pnl_segment.space.ref_space import RefSpace
from scipy.ndimage import binary_dilation


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
    >>> [1, 1] in m1
    True
    >>> (2, 0) in m1
    False
    >>> list(m1)
    [(0, 0), (1, 1), (2, 2)]
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
    >>> m2.negate().x
    array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]])
    """

    def __iter__(self):
        return iter(tuple(ijk) for ijk in np.vstack(np.where(self.x > 0)).T)

    def iter_multidim(self, n):
        for ijk in self:
            for _n in range(n):
                yield (*ijk, _n)

    def __contains__(self, ijk):
        return bool(self.x[tuple(ijk)])

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return np.sum((self.x > 0).flatten())

    def __init__(self, x, ref_space=None):
        self.x = x.astype(bool)
        self.ref_space = ref_space

    def negate(self):
        return Mask(np.logical_not(self.x), ref_space=self.ref_space)

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
    def from_img(img, fnc_include=None):
        # defaults to include all positive values in mask
        if fnc_include is None:
            def is_positive(x):
                return x > 0

            fnc_include = is_positive

        ref = RefSpace(affine=img.affine, shape=img.shape)
        return Mask(fnc_include(img.get_data()), ref_space=ref)

    @staticmethod
    def from_nii(f_nii, **kwargs):
        return Mask.from_img(nib.load(str(f_nii)), **kwargs)

    def to_nii(self, f_out=None, f_ref=None):
        # get f_out
        if f_out is None:
            _, f_out = tempfile.mkstemp(suffix='.nii.gz')

        # get ref
        if f_ref:
            ref = RefSpace.from_nii(f_ref)
        else:
            ref = self.ref_space

        # save
        img = nib.Nifti1Image(self.x.astype(np.int8), affine=ref.affine)
        img.to_filename(str(f_out))

        return f_out

    def insert(self, data_img, x, add=False):
        """ inserts x into data_img where mask is

        Args:
            data_img (np.array): base image
            x (np.array): points to be added
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

        if len(x.flatten()) != len(self) * d:
            raise AttributeError('size mismatch, data len does not match mask')

        # insert given data
        for _x, ijk in zip(x.flatten(), ijk_iter):
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

        return data[self.x.astype(bool)]

    def apply_from_nii(self, f_nii):
        img = nib.load(str(f_nii))
        if self.ref_space is not None and \
                not np.array_equal(img.affine, self.ref_space.affine):
            raise AttributeError('affine mismatch')

        return self.apply(img.get_data())

    def dilate(self, r):
        x = binary_dilation(self.x, iterations=r)
        return Mask(x, ref_space=self.ref_space)

    @staticmethod
    def build_intersection(mask_list, thresh=1):
        """ builds intersection of a set of masks

        Args:
            mask_list (list): list of masks
            thresh (float): in (0, 1], how many of masks true @ ijk to include

        Returns:
            mask (Mask): intersection of masks
        """
        # get 'intersection'
        c = len(mask_list) * thresh
        x_all = sum(m.x.astype(bool) for m in mask_list) >= c

        # if all ref_space are same, keep it, otherwise discard
        ref_space = mask_list[0].ref_space
        for m in mask_list[1:]:
            if m.ref_space != ref_space:
                ref_space = None
                break

        return Mask(x_all, ref_space=ref_space)

    @staticmethod
    def build_intersection_from_nii(f_nii_list, **kwargs):
        mask_list = [Mask.from_nii(f) for f in f_nii_list]
        return Mask.build_intersection(mask_list, **kwargs)
