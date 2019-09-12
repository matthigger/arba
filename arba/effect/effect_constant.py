import pathlib
import tempfile
from copy import copy

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.stats import multivariate_normal

from arba.region import FeatStat
from arba.space import Mask
from .effect import Effect


def draw_random_u(d):
    """ Draws random vector in d dimensional unit sphere

    Args:
        d (int): dimensionality of vector
    Returns:
        u (np.array): random vector in d dim unit sphere
    """
    mu = np.zeros(d)
    cov = np.eye(d)
    u = multivariate_normal.rvs(mean=mu, cov=cov)
    return u / np.linalg.norm(u)


class EffectConstant(Effect):
    """ an effect is a constant offset to a set of voxels

    Attributes:
        offset (np.array): average offset of effect on a voxel
        eff_img (np.array): offset image (memoized)
        u (np.array): offset direction, normalized
        t2 (float): t squared distance
    """

    def __init__(self, offset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = np.atleast_1d(offset).astype(float)
        self._eff_img = None

    def apply(self, x, negate=False):
        """ given an image, feat, applies the effect
        """
        if negate:
            return x - self.eff_img
        else:
            return x + self.eff_img

    @staticmethod
    def from_fs_t2(fs, t2, mask, edge_n=None, u=None):
        """ scales effect with observations

        Args:
            fs (FeatStat): stats of affected area
            t2 (float): ratio of effect to population variance
            mask (Mask): effect location
            edge_n (int): number of voxels on edge of mask (taxicab erosion)
                          which have a 'scaled' effect.  For example, if edge_n
                          = 1, then the outermost layer of voxels has only half
                          the offset applied.  see Effect.scale and eff_img for
                          detail
            u (array): direction of offset
        """

        if t2 < 0:
            raise AttributeError('t2 must be positive')

        # get direction u
        if u is None:
            u = draw_random_u(d=fs.d)
        elif len(u) != fs.d:
            raise AttributeError('direction offset must have same len as fs.d')

        # compute scale
        if edge_n is None:
            scale = mask
        else:
            scale = distance_transform_cdt(mask,
                                           metric='taxicab') / (edge_n + 1)
            scale[scale >= 1] = 1

        # build effect with proper direction, scale to proper t2
        # (ensure u is copied so we have a spare to validate against)
        eff = Effect(mask=mask, offset=copy(u), fs=fs, scale=scale)
        eff.t2 = t2

        u = np.atleast_1d(u).astype(float)
        u *= 1 / np.linalg.norm(u)
        assert np.allclose(eff.u, u), 'direction error'
        assert np.allclose(eff.t2, t2), 't2 scale error'

        return eff

    @property
    def t2(self):
        if self.fs is None:
            return None
        return self.offset @ self.fs.cov_inv @ self.offset

    @t2.setter
    def t2(self, val):
        """ change scale of effect to achieve new t2
        """
        if self.fs is None:
            raise AttributeError('fs required to set t2')
        self._eff_img = None
        self.offset *= np.sqrt(float(val) / self.t2)

    @property
    def u(self):
        return self.offset / np.linalg.norm(self.offset)

    @u.setter
    def u(self, val):
        """ changes direction of effect, keeps t2 constant
        """
        if self.fs is None:
            raise AttributeError('fs required to set u')
        self._eff_img = None
        c = val @ self.fs.cov_inv @ val
        self.offset = np.atleast_1d(val) * self.t2 / c

    def to_nii(self, f_out=None):
        if f_out is None:
            f_out = tempfile.NamedTemporaryFile(suffix=f'_effect.nii.gz').name
            f_out = pathlib.Path(f_out)

        img = nib.Nifti1Image(self.eff_img, affine=self.mask.ref.affine)
        img.to_filename(str(f_out))

        return f_out

    @property
    def eff_img(self):
        if self._eff_img is None:
            shape = (*self.mask.shape, self.d)
            self._eff_img = np.zeros(shape)
            for idx in range(self.d):
                self._eff_img[..., idx] = self.offset[idx] * self.scale
        return self._eff_img

    @property
    def d(self):
        return len(self.offset)
