import numpy as np
from scipy.stats import f

from .feat_stat import FeatStat, FeatStatSingle, FeatStatEmpty
from .reg import Region


class RegionT2(Region):
    """ computes MSE error of representing T-sq per voxel by a single value

    Attributes:
        t2_fs (FeatStat): describes set of t2 stats per voxel
    """

    def __init__(self, *args, t2_per_vox=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._t2 = None
        self._pval = None
        self._sq_error = None

        if len(self) == 1:
            self.t2_fs = FeatStatSingle(mu=self.t2)
        elif t2_per_vox is None:
            self.t2_fs = FeatStatEmpty()
        else:
            assert len(t2_per_vox) == len(self), 't2_per_vox mismatch'
            self.t2_fs = FeatStat.from_array(t2_per_vox,
                                             obs_greater_dim=True)

    @property
    def t2(self):
        # memoize
        if self._t2 is None:
            self._t2 = self.get_t2()
        if self._t2 < 0:
            raise AttributeError('invalid t2')
        return self._t2

    def get_t2(self):
        """ t squared distance between groups

        t2(pop_1, pop_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the pooled covariance and u_i is mean in population_i
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu

        # note: all feat_stat.cov are simple averages (np.cov(x, ddof=0))
        cov = (fs0.n * fs0.cov +
               fs1.n * fs1.cov) / (fs0.n + fs1.n)

        return mu_diff @ np.linalg.inv(cov) @ mu_diff

    @property
    def pval(self):
        if self._pval is None:
            self._pval = self.get_pval()
        if np.isnan(self._pval):
            raise AttributeError('invalid pval')
        return self._pval

    def get_pval(self):
        """ gets pvalue of t squared stat

        see http://math.bme.hu/~marib/tobbvalt/tv5.pdf
        """
        # dimensionality of data
        p = next(iter(self.fs_dict.values())).d

        # number of observations per group
        n, m = [fs.n for fs in self.fs_dict.values()]

        # compute f stat
        f_stat = self.t2 * n * m / (n + m)
        f_stat *= (n + m - p - 1) / (p * (n + m - 2))

        # compute pval
        pval = f.sf(f_stat, dfd=p, dfn=n + m - p - 1)

        return pval

    @property
    def sq_error(self):
        """ squared error of t2 per voxel represented as whole region t2

        define:

        delta = self.t2 - self.t2_fs.mu

        then, let x_i represent the t2 stat at voxel i

        sq_error = sum_i (x_i - self.t2)^2
                 = sum_i (x_i - (self.t2_fs.mu + delta))^2
                 = ... complete the square ...
                 = self.t2_fs.n * (self.t2_fs.cov + delta ^ 2)

        rather than store voxel wise t2 stats to compute sq_error, we store
        their summary statistics n, mu and cov.  above equality shows how
        sufficient to get sq_error
        """
        if self._sq_error is None:
            delta = self.t2_fs.mu - self.t2
            self._sq_error = self.t2_fs.n * (self.t2_fs.cov[0, 0] + delta ** 2)
        return self._sq_error

    @staticmethod
    def get_error(reg_1, reg_2, reg_union=None):
        if reg_union is None:
            reg_union = reg_1 + reg_2
        return reg_union.sq_error - reg_1.sq_error - reg_2.sq_error

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            reg_out = super().__add__(other)
            reg_out.t2_fs = self.t2_fs
            return reg_out

        if not isinstance(other, type(self)):
            raise TypeError

        reg_out = super().__add__(other)
        reg_out.t2_fs = self.t2_fs + other.t2_fs

        return reg_out

    __radd__ = __add__

    def reset(self):
        self._t2 = None
        self._pval = None
