import os
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from arba.space.ref_space import get_ref


class Mask(np.ndarray):
    """ boolean array which preserves reference spacing
    """

    @staticmethod
    def from_img(img):
        ref = get_ref(img)
        return Mask(img.get_data(), ref=ref)

    @staticmethod
    def from_nii(f_nii):
        return Mask.from_img(nib.load(str(f_nii)))

    def __new__(cls, input_array, ref=None):
        # https://docs.scipy.org/doc/numpy-1.15.0/user/basics.subclassing.html
        obj = np.asarray(input_array).astype(bool).view(cls)
        obj.ref = get_ref(ref)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ref = getattr(obj, 'ref', None)

    def __len__(self):
        return np.sum((self).flatten())

    def to_nii(self, f_out=None, ref=None):
        # get f_out
        if f_out is None:
            f, f_out = tempfile.mkstemp(suffix='.nii.gz')
            os.close(f)

        # get ref
        ref = get_ref(ref)
        if ref is None:
            ref = self.ref

        # save
        # todo: how to output as bool type (not uint8)?
        img = nib.Nifti1Image(self.astype(np.uint8), affine=ref.affine)
        img.to_filename(str(f_out))

        return f_out

    def __eq__(self, other):
        if self.ref != other.ref:
            return False
        return not np.logical_xor(self, other).any()

    def dilate(self, r):
        x = binary_dilation(self, iterations=r)
        return Mask(x, ref=self.ref)

    # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
    def __reduce__(self):
        pickled_state = super(Mask, self).__reduce__()
        new_state = pickled_state[2] + (self.ref,)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        self.ref = state[-1]
        super(Mask, self).__setstate__(state[0:-1])
