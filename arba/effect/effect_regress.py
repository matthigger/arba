import numpy as np
from scipy.optimize import minimize_scalar

from .effect import Effect
from .get_r2 import get_r2


class EffectRegress(Effect):
    """ a consant offset depending on linear fnc of subject features:

    feat_img_offset = feat @ beta

    Attributes:
        beta (np.array): (dim_sbj feat dim_img) mapping from sbj to img feat
    """

    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta

    @staticmethod
    def from_r2(r2, feat_img, feat_sbj, img_pool_cov=None, contrast=None,
                *args, **kwargs):
        """ yields an effect which achieve r2

        NOTE: the direction of the effect (beta) is chosen as the min MSE
        direction already in feat_img & data_sbj

        Args:
            r2 (float): target r2
            feat_img (np.array): (num_sbj, dim_img) image features
            feat_sbj (np.array): (num_sbj, dim_sbj) subject features
            img_pool_cov (np.array): the pooled (across sbj) covariance of
                                     imaging features.  if not passed then it
                                     is assumed 0
            contrast (np.array): (dim_sbj) contrast[i] = 1 => sbj feat i is not
                                 a nuisance feature

        Returns:
            eff (EffectRegress)
        """

        assert 0 <= r2 < 1, 'invalid r2'

        if img_pool_cov is None:
            dim_img = feat_img.shape[1]
            img_pool_cov = np.zeros((dim_img, dim_img))

        beta = np.linalg.pinv(feat_sbj) @ feat_img

        def fnc(scale):
            """ computes r2 under scale factor, returns error to target r2
            """
            _beta = beta * scale
            _r2, _ = get_r2(beta=_beta,
                            x=feat_sbj,
                            y=feat_img + feat_sbj @ _beta,
                            y_pool_cov=img_pool_cov,
                            contrast=contrast)
            return (_r2 - r2) ** 2

        res = minimize_scalar(fnc)
        assert fnc(res.x) < 1e-7, 'optimization error'

        beta *= res.x

        return EffectRegress(*args, beta=beta, **kwargs)

    def get_offset_array(self, feat_sbj):
        """ gets array, in shape of self.mask.ref, of effect

        Args:
            feat_sbj (np.array): (num_sbj, num_sfeat) subject features

        Returns:
            eff_delta (np.array): (space0, space1, space2, num_sbj, dim_img)
                                  offset (zero outside mask)
        """
        dim_img = self.beta.shape[1]
        shape = self.mask.shape + (feat_sbj.shape[0], dim_img)
        eff_delta = np.zeros(shape)

        for sbj_idx, delta in enumerate(feat_sbj @ self.beta):
            eff_delta[self.mask, sbj_idx, ...] += delta

        return eff_delta
