import numpy as np

from arba.simulate.effect.effect import Effect


class EffectRegress(Effect):
    """ a consant offset, depending on linear fnc of 'subject' features

    img_feat_offset = sbj_feat @ beta

    we distinguish between 'subject' features, those associated with a whole
    image (e.g. age, sex) and 'image' features, those associated with a
    specific voxel in a specific image (e.g. FA, MD).

    'cov' refers to the total feature covariance.  'eps' specifically refers to
    the covariance of the 'error' (covariance unrelated to the regression)

    Attributes:
        beta (np.array): (dim_sbj x dim_img) mapping from sbj to img feat
        cov_sbj (np.array): covariance of sbj features
        cov_img (np.array): covariance of img features (after effect applied)
        eps_img (np.array): covariance of img features error
        r2 (float): the r squared value
        feat_mapper (fnc): maps sbj to features
    """

    @staticmethod
    def from_r2(r2, eps_img, cov_sbj, u=None, *args, **kwargs):
        """
        NOTE: the target r2 is approximate.  the offsets differ per sbj.  as a
        result the cov_sbj observed will change under the effect
        """

        assert 0 <= r2 <= 1, 'invalid r2'
        assert r2 != 1 or np.trace(eps_img) == 0, 'eps_img must be 0 if r2=1'

        cov_sbj = np.atleast_2d(cov_sbj)
        eps_img = np.atleast_2d(eps_img)

        if u is None:
            u = np.ones(shape=(cov_sbj.shape[0], eps_img.shape[0]))

        tr_cov_img = np.trace(eps_img) / (1 - r2)
        tr_cov_img_regress = tr_cov_img - np.trace(eps_img)
        _tr_cov_img_regress = np.trace(u.T @ cov_sbj @ u)
        scale = (tr_cov_img_regress / _tr_cov_img_regress) ** .5
        beta = u * scale

        return EffectRegress(beta=beta, cov_sbj=cov_sbj, eps_img=eps_img,
                             *args, **kwargs)

    def __init__(self, beta, cov_sbj=None, eps_img=None, feat_mapper=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.feat_mapper = feat_mapper

        self.cov_sbj = np.atleast_2d(cov_sbj)
        self.eps_img = np.atleast_2d(eps_img)
        if self.fs is not None:
            if self.eps_img is None:
                self.eps_img = self.fs.cov
            else:
                assert np.isclose(self.fs.cov, eps_img), \
                    'eps_img double specified'

        # compute r2 and cov_img
        self.cov_sbj = cov_sbj
        self.r2 = None
        if (self.cov_sbj is not None) and (self.eps_img is not None):
            self.cov_img = self.eps_img + \
                           self.beta.T @ self.cov_sbj @ self.beta

            self.r2 = 1 - np.trace(self.eps_img) / np.trace(self.cov_img)

    def project(self, sbj=None, feat_sbj=None):
        assert (sbj is None) != (feat_sbj is None), \
            'either sbj xor feat_sbj required'

        if sbj is not None:
            assert self.feat_mapper is not None, \
                'feat_mapper required in __init__'
            feat_sbj = self.feat_mapper(sbj)

        return feat_sbj @ self.beta

    def get_offset_array(self, sbj_list):
        """ gets array, in shape of self.mask.ref, of effect

        Args:
            sbj_list (list): list of subjects

        Returns:
            eff_delta (np.array): (space0, space1, space2, sbj_idx, feat_idx)
                                  offset (zero outside mask)
        """

        shape = self.mask.shape + (len(sbj_list), self.eps_img.shape[0])
        eff_delta = np.zeros(shape)

        for sbj_idx, sbj in enumerate(sbj_list):
            sbj_delta = self.project(sbj=sbj)
            eff_delta[self.mask, sbj_idx, ...] += sbj_delta

        return eff_delta


if __name__ == '__main__':
    dim_sbj = 2
    dim_img = 3

    beta = np.ones(shape=(dim_sbj, dim_img))
    mask = np.ones(shape=(4, 4, 4))
    cov_sbj = np.eye(dim_sbj)
    eps_img = np.eye(dim_img)

    r0 = EffectRegress(beta=np.ones(shape=(dim_sbj, dim_img)),
                       mask=mask,
                       cov_sbj=cov_sbj,
                       eps_img=eps_img)

    r1 = EffectRegress.from_r2(r2=.9,
                               mask=mask,
                               eps_img=eps_img,
                               cov_sbj=cov_sbj)
