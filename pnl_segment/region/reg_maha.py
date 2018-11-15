from scipy.stats import chi2

from .reg_kl import RegionKL


class RegionMaha(RegionKL):
    @property
    def maha(self):
        return self._obj

    def get_obj(self):
        """ negative Mahalanobis squared between active_grp

        (assumes each distribution is normal and covar are equal).  Note that
        if covar between active_grp are equal then this is equivalent (and
        faster) than RegionMaxKL

        maha(p_1, p_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the common covariance / region size (in voxels)
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu
        fs_sum = fs0 + fs1
        return mu_diff @ fs_sum.cov_inv @ mu_diff

    @property
    def pval(self):
        d = next(iter(self.fs_dict.values())).d
        return chi2.sf(self.maha * len(self), df=d)