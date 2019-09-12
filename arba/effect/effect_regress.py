import numpy as np
from scipy.optimize import minimize_scalar

from arba.effect import Effect, compute_r2


class EffectRegress(Effect):
    """ a consant offset depending on linear fnc of subject features:

    img_feat_offset = data_sbj.x @ beta

    Attributes:
        beta (np.array): (dim_sbj x dim_img) mapping from sbj to img feat
    """

    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta

    def compute_r2(self, *args, **kwargs):
        return compute_r2(*args, beta=self.beta, **kwargs)

    @staticmethod
    def from_r2(r2, img_feat, sbj_feat, img_pool_cov=None, *args, **kwargs):
        """ yields an effect which achieve r2

        NOTE: the direction of the effect (beta) is chosen as the min MSE
        direction already in img_feat & data_sbj

        Args:
            r2 (float): target r2
            img_feat (np.array): (num_sbj, dim_img) image features
            sbj_feat (np.array): (num_sbj, dim_sbj) subject features
            img_pool_cov (np.array): the pooled (across sbj) covariance of
                                     imaging features.  if not passed then it
                                     is assumed 0

        Returns:
            eff (EffectRegress)
        """

        assert 0 <= r2 < 1, 'invalid r2'

        if img_pool_cov is None:
            dim_img = img_feat.shape[1]
            img_pool_cov = np.zeros((dim_img, dim_img))

        beta = sbj_feat.pseudo_inv @ img_feat

        def fnc(scale):
            """ computes r2 under scale factor, returns error to target r2
            """
            _beta = beta * scale
            _r2 = compute_r2(_beta,
                             x=sbj_feat.x,
                             y=img_feat + sbj_feat.x @ _beta,
                             y_pool_cov=img_pool_cov)
            return (_r2 - r2) ** 2

        res = minimize_scalar(fnc)
        assert fnc(res.x) < 1e-7, 'optimization error'

        beta *= res.x

        return EffectRegress(*args, beta=beta, **kwargs)

    def get_offset_array(self, sbj_feat):
        """ gets array, in shape of self.mask.ref, of effect

        Args:
            sbj_feat (DataSubject):

        Returns:
            eff_delta (np.array): (space0, space1, space2, num_sbj, dim_img)
                                  offset (zero outside mask)
        """
        dim_img = self.beta.shape[1]
        shape = self.mask.shape + (sbj_feat.num_sbj, dim_img)
        eff_delta = np.zeros(shape)

        for sbj_idx, delta in enumerate(sbj_feat.x @ self.beta):
            eff_delta[self.mask, sbj_idx, ...] += delta

        return eff_delta
