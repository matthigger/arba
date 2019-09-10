import numpy as np
from scipy.optimize import minimize_scalar

from arba.effect import Effect


def compute_r2(beta, img_feat, sbj_feat, img_pool_cov=None):
    """ computes r2

    Args:
        beta (np.array):
        img_feat (np.array): (num_sbj, dim_img) imaging features
        sbj_feat (np.array): (num_sbj, dim_sbj) subject features
        img_pool_cov (np.array): pooled (across sbj) of imaging feature cov
                                 across region

    Returns:
        r2 (float): coefficient of determination
    """
    # get tr_pool_cov
    if img_pool_cov is None:
        tr_pool_cov = 0
    else:
        tr_pool_cov = np.trace(img_pool_cov)

    # compute error
    delta = img_feat - sbj_feat @ beta

    # compute eps, covariance of error
    eps = np.atleast_2d((delta.T @ delta) / delta.shape[0])

    # compute covariance of imaging features
    cov = np.atleast_2d(np.cov(img_feat.T, ddof=0))

    # comptue r2
    r2 = 1 - (np.trace(eps) + tr_pool_cov) / (np.trace(cov) + tr_pool_cov)

    return r2


class EffectRegress(Effect):
    """ a consant offset depending on linear fnc of subject features

    img_feat_offset = sbj_feat @ beta

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
        direction already in img_feat & sbj_feat

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

        beta = np.linalg.pinv(sbj_feat) @ img_feat

        def fnc(scale):
            """ computes r2 under scale factor, returns error to target r2
            """
            _beta = beta*scale
            _r2 = compute_r2(_beta,
                             img_feat=img_feat + sbj_feat @ _beta,
                             sbj_feat=sbj_feat,
                             img_pool_cov=img_pool_cov)
            return (_r2 - r2) ** 2

        res = minimize_scalar(fnc)
        assert res.success, 'optimization error'

        beta *= res.x

        return EffectRegress(*args, beta=beta, **kwargs)

    def get_offset_array(self, sbj_feat):
        """ gets array, in shape of self.mask.ref, of effect

        Args:
            sbj_feat (np.array): (num_sbj, dim_sbj) subject features

        Returns:
            eff_delta (np.array): (space0, space1, space2, num_sbj, dim_img)
                                  offset (zero outside mask)
        """
        num_sbj = sbj_feat.shape[0]
        dim_img = self.beta.shape[1]
        shape = self.mask.shape + (num_sbj, dim_img)
        eff_delta = np.zeros(shape)

        for sbj_idx, delta in enumerate(self.beta @ sbj_feat):
            eff_delta[self.mask, sbj_idx, ...] += delta

        raise NotImplementedError('check')

        return eff_delta
