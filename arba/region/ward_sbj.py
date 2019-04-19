import numpy as np

from .ward_grp import RegionWardGrp


class RegionWardSbj(RegionWardGrp):
    """ computes t2 according to split covariance model (across sbj and space)
    """

    def __init__(self, *args, sig_sbj, **kwargs):
        super().__init__(*args, **kwargs)
        self.sig_sbj = sig_sbj
        self.sig_space = self.cov_pooled - sig_sbj

        assert (self.sig_space >= 0).all(), 'invalid sig_sbj passed'

    def get_t2(self):
        """ hotelling t squared distance between groups

        (n0 * n1) / (n0 + n1) (u_1 - u_0)^T sig^-1 (u_1 - u_0)

        where sig = sig_sbj + sig_space / len(self)
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu
        sig = self.sig_sbj + self.sig_space / len(self.pc_ijk)
        quad_term = mu_diff @ np.linalg.inv(sig) @ mu_diff

        scale = (fs0.n * fs1.n) / (fs0.n + fs1.n)

        t2 = scale * quad_term

        if t2 < 0:
            raise AttributeError('invalid t2')

        return t2
