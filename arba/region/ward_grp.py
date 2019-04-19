import numpy as np
from scipy.stats import f

from .reg import Region


class RegionWardGrp(Region):
    """ region object, segmentations constructed to cluster observations

    sq_error_tr: identify segmention which minimizes
        sum_region sum_i ||x_i - mu_{region, group}||^2
    where mu_{region, group} is the average observation of the group (e.g. Ill
    or Healthy) within some region.

    sq_error_det: identify segmentation which maximizes likelihood under normal
    model per group
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t2 = None
        self._pval = None

        # note: all feat_stat.cov are simple averages (np.cov(x, ddof=0)),
        # this is critical for computation of sq_error_det
        fs0, fs1 = self.fs_dict.values()
        self.cov_pooled = (fs0.n * fs0.cov +
                           fs1.n * fs1.cov) / (fs0.n + fs1.n)

    @property
    def pval(self):
        if self._pval is None:
            self._pval = self.get_pval()
        if np.isnan(self._pval):
            raise AttributeError('invalid pval')
        return self._pval

    @property
    def t2(self):
        # memoize
        if self._t2 is None:
            self._t2 = self.get_t2()
            if self._t2 < 0:
                raise AttributeError('invalid t2')
        return self._t2

    def get_t2(self):
        """ (unweighted) t squared distance between groups

        t2(pop_1, pop_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the pooled covariance and u_i is mean in population_i
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu

        return mu_diff @ np.linalg.inv(self.cov_pooled) @ mu_diff

    def get_pval(self):
        """ gets pvalue of t squared stat

        see http://math.bme.hu/~marib/tobbvalt/tv5.pdf
        """
        # dimensionality of data
        p = next(iter(self.fs_dict.values())).d

        # number of observations per group
        n, m = [fs.n for fs in self.fs_dict.values()]

        # # number of observations per group is number of voxels
        # # kludge: invalid pval if n=m=1, f_stat diverges with 0 in denominator
        # n = max(len(self), 2)
        # m = n

        # compute f stat
        f_stat = self.t2 * n * m / (n + m)
        f_stat *= (n + m - p - 1) / (p * (n + m - 2))

        # compute pval
        pval = f.sf(f_stat, dfd=p, dfn=n + m - p - 1)

        return pval

    @property
    def sq_error_det(self):
        """ the weighted sum of the determinants of pooled covar matrices

        let us index an observation by i, so we have {x_i, r_i, omega_i} where
        x_i is the feature vector, r_i is the region it belongs to and omega_i
        is the group (e.g. Schizophrenia or Healthy)

        segmentation_ML = arg max P({x_i|r_i, omega_i}_i)
                        = ...independence, normality, equal grp covariance ...
                        = sum_r N_r |cov_pooled_r|

        so we define:

        sq_error_det = N_r |cov_pooled_r|

        where N_r is number of observations in region r.
        """

        n = sum(fs.n for fs in self.fs_dict.values())
        return np.linalg.det(self.cov_pooled) * n

    @property
    def sq_error_tr(self):
        """ the weighted sum of the trace of pooled covar matrices

        (see doc in sq_error_det for notation)

        sq_error_tr = sum_i || x_i - mu_{r_i, omega_i}||^2
                    = ...
                    = sum_r  N_r tr(cov_pooled_r)
        """

        n = sum(fs.n for fs in self.fs_dict.values())
        return np.trace(self.cov_pooled) * n

    @staticmethod
    def get_error_det(reg_0, reg_1, reg_union=None):
        if reg_union is None:
            reg_union = reg_0 + reg_1
        return reg_union.sq_error_det - reg_0.sq_error_det - reg_1.sq_error_det

    @staticmethod
    def get_error_tr(reg_0, reg_1, reg_union=None):
        if reg_union is None:
            reg_union = reg_0 + reg_1
        return reg_union.sq_error_tr - reg_0.sq_error_tr - reg_1.sq_error_tr

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            reg_out = super().__add__(other)
            return reg_out

        if not isinstance(other, type(self)):
            raise TypeError

        reg_out = super().__add__(other)

        return reg_out

    __radd__ = __add__

    def reset(self):
        self._t2 = None
        self._pval = None
