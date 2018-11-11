import pathlib

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine


class RefSpace:
    """ maps between array space and xyz scanner space"""

    @staticmethod
    def from_nii(f_nii):
        img_ref = nib.load(str(f_nii))
        return RefSpace(affine=img_ref.affine, shape=img_ref.shape)

    @staticmethod
    def from_trk(f_trk, validate=True):
        affine = nib.streamlines.load(str(f_trk)).affine
        if validate:
            x = affine[:3, :3]
            if np.allclose(x @ x.T, np.eye(3)):
                # we enforce orthonormal affines for xyz, can't think of a use
                # case for scaling and it may often be the case that ijk
                # affines are stored in trk files erroneously
                raise AttributeError('affine is not orthonormal for trk file')
        return RefSpace(affine)

    def __init__(self, affine=np.eye(4), shape=None):
        self.affine = affine
        self.affine_inv = np.linalg.inv(self.affine)
        self.shape = shape

    def __str__(self):
        return f'RefSpace(affine={self.affine}, shape={self.shape})'

    def to_rasmm(self, x):
        """ maps from ijk space to scanner xyz """
        return apply_affine(self.affine, x)

    def from_rasmm(self, xyz, round_flag=False):
        """ maps from scanner xyz into array, lossy if round=True"""
        x = apply_affine(self.affine_inv, xyz)
        if round_flag:
            x = np.rint(x).astype(int)
        return x

    def __eq__(self, other):
        if not np.allclose(self.affine, other.affine):
            return False

        if self.shape is None and other.shape is None:
            return True

        return np.array_equal(self.shape, other.shape)


def get_ref(ref):
    """ convenience: accepts path to trk, nii or a RefSpace.  returns RefSpace
    """
    if ref is None:
        return None

    if not isinstance(ref, RefSpace):
        f_ref = pathlib.Path(ref)
        if '.trk' in f_ref.suffixes:
            ref = RefSpace.from_trk(f_ref)
        elif '.nii' in f_ref.suffixes:
            ref = RefSpace.from_nii(f_ref)
        else:
            raise AttributeError('ref not recognized')

    return ref
