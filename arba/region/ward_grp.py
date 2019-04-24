from collections import defaultdict

import numpy as np
from scipy.stats import f

from .pool_cov import PoolCov
from .reg import Region


class RegionWardGrp(Region):
    """ region object, segmentations constructed to cluster observations

    sq_error_det -> max_likelihood
    sq_error_tr -> min distance between observation and population means

    (see methods for detail)
    """

    @classmethod
    def from_data(cls, pc_ijk, file_tree, split, **kwargs):
        # get stats across region per grp
        fs_dict = defaultdict(list)
        for ijk in pc_ijk:
            for grp, sbj_list in split.items():
                fs_dict[grp].append(file_tree.get_fs(ijk, sbj_list=sbj_list))
        fs_dict = {k: sum(l) for k, l in fs_dict.items()}

        return cls(pc_ijk=pc_ijk, fs_dict=fs_dict, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t2 = None
        self._pval = None

        self.cov_pool = PoolCov.sum([PoolCov.from_feat_stat(fs)
                                     for fs in self.fs_dict.values()])

    def reset(self):
        self._t2 = None
        self._pval = None

    @property
    def pval(self):
        # memoize
        if self._pval is None:
            self._pval = self.get_pval()
        return self._pval

    @property
    def t2(self):
        # memoize
        if self._t2 is None:
            self._t2 = self.get_t2()
        return self._t2

    def get_t2(self):
        """ hotelling t squared distance between groups

        (n0 * n1) / (n0 + n1) (u_1 - u_0)^T cov_pool^-1 (u_1 - u_0)
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu
        quad_term = mu_diff @ np.linalg.inv(self.cov_pool.value) @ mu_diff

        n0, n1 = self.n0n1
        scale = (n0 * n1) / (n0 + n1)

        t2 = scale * quad_term

        if t2 < 0:
            raise AttributeError('invalid t2')

        return t2

    @property
    def n0n1(self):
        fs0, fs1 = self.fs_dict.values()
        return fs0.n, fs1.n

    def get_pval(self):
        """ gets pvalue of t squared stat

        see http://math.bme.hu/~marib/tobbvalt/tv5.pdf
        """
        # dimensionality of data
        p = next(iter(self.fs_dict.values())).d

        # compute scale to f stat
        n0, n1 = self.n0n1
        f_scale = (n0 + n1 - p - 1) / (p * (n0 + n1 - 2))

        # compute pval
        pval = f.sf(self.t2 * f_scale, dfd=p, dfn=n0 + n1 - p - 1)

        if np.isnan(pval):
            raise AttributeError('invalid pval')

        return pval

    @property
    def sq_error_det(self):
        """ the weighted sum of the determinants of pooled covar matrices

        we have {x_i, r_i, omega_i} where x_i is the feature vector, r_i is the
        region it belongs to and omega_i is the group (e.g. Schizophrenia or
        Healthy)

        segmentation_ML = arg max P({x_i|r_i, omega_i}_i)
                        = ...independence, normality, equal grp covariance ...
                        = sum_r N_r |cov_pooled_r|

        so we define:

        sq_error_det = N_r |cov_pooled_r|

        where N_r is number of observations in region r.
        """

        n = sum(fs.n for fs in self.fs_dict.values())
        return np.linalg.det(self.cov_pool.value) * n

    @property
    def sq_error_tr(self):
        """ the weighted sum of the trace of pooled covar matrices

        (see doc in sq_error_det for notation)

        sq_error_tr = sum_i || x_i - mu_{r_i, omega_i}||^2
                    = ...
                    = sum_r  N_r tr(cov_pooled_r)
        """

        n = sum(fs.n for fs in self.fs_dict.values())
        return np.trace(self.cov_pool.value) * n

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
