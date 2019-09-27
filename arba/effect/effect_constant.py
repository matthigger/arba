import pathlib
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.stats import multivariate_normal

from arba.region import get_t2_stats
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
    """ an effect is a constant delta to a set of voxels

    Attributes:
        delta (np.array): feature offset applied to each voxel
        pool_cov (np.arry): population pooled covar
        t2 (float): hotelling's t squared
        eff_img (np.array): delta image (memoized)
    """

    def __init__(self, delta, pool_cov=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = np.atleast_1d(delta).astype(float)
        self.pool_cov = pool_cov
        self.t2 = self.delta @ np.linalg.inv(self.pool_cov) @ self.delta
        self._eff_img = None

    def apply(self, x, negate=False):
        """ given an image, feat, applies the effect
        """
        if negate:
            return x - self.eff_img
        else:
            return x + self.eff_img

    @staticmethod
    def from_t2(t2, mask, data_img, grp_sbj_list_dict, edge_n=None,
                grp_target=None):
        """ scales effect with observations

        Args:
            t2 (float): ratio of effect to population variance
            mask (Mask): effect location
            data_img (DataImage):
            grp_target: grp that effect is applied to
            grp_sbj_list_dict (dict): keys are grp labels, values are sbj_list
            edge_n (int): number of voxels on edge of mask (taxicab erosion)
                          which have a 'scaled' effect.  For example, if edge_n
                          = 1, then the outermost layer of voxels has only half
                          the delta applied.  see Effect.scale and eff_img for
                          detail
        """
        # get grp_tuple, grp_tuple[1] has effect applied
        assert len(grp_sbj_list_dict) == 2, 'invalid grp_sbj_list_dict'
        grp_tuple = sorted(grp_sbj_list_dict.keys())
        if grp_target is not None:
            assert grp_target in grp_tuple
            if grp_target is not grp_tuple[1]:
                grp_tuple = reversed(grp_tuple)

        # get fs_dict
        fs_dict = dict()
        for grp, sbj_list in grp_sbj_list_dict.items():
            fs_dict[grp] = data_img.get_fs(sbj_list=sbj_list, mask=mask)

        delta, cov_pool, _ = get_t2_stats(fs_dict, grp_tuple=grp_tuple)

        # compute scale
        if edge_n is None:
            scale = mask
        else:
            scale = distance_transform_cdt(mask,
                                           metric='taxicab') / (edge_n + 1)
            scale[scale >= 1] = 1

        # build effect with proper direction
        # todo: refactor all pool_cov -> cov_pool
        eff = EffectConstant(mask=mask, delta=delta, pool_cov=cov_pool,
                             scale=scale)

        # scale to given t2
        return eff.to_t2(t2)

    def to_t2(self, t2):
        """ returns an EffectConstant whose delta is scaled to yield some t2

        Args:
            t2 (float): desired t2 value
        """
        assert t2 >= 0, 't2 must be positive'

        delta = np.sqrt(t2 / self.t2) * self.delta

        eff = EffectConstant(delta, pool_cov=self.pool_cov, mask=self.mask,
                             scale=self.scale)

        assert np.isclose(t2, eff.t2), 't2 scale error'

        return eff

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
                self._eff_img[..., idx] = self.delta[idx] * self.scale
        return self._eff_img
