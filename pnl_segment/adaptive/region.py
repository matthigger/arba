from itertools import permutations

import numpy as np


class Region:
    """ a set of voxels and their associated features

    Attributes:
        pc_ijk (PointCloudIJK)
        feat_stat (dict): contains feat stats for img sets of different grps
    """
    # toggles whether obj should be maximized or minimized
    max_flag = False

    @property
    def obj(self):
        # memoize
        if self._obj is None:
            self._obj = self.get_obj()
        return self._obj

    def __init__(self, pc_ijk, feat_stat, active_grp=None):
        self.pc_ijk = pc_ijk
        self.feat_stat = feat_stat
        self._obj = None
        self.active_grp = active_grp

    def __str__(self):
        return f'{self.__class__} with {self.feat_stat}'

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.feat_stat)

        feat_stat = {grp: self.feat_stat[grp] + other.feat_stat[grp]
                     for grp in self.feat_stat.keys()}

        return type(self)(pc_ijk=self.pc_ijk + other.pc_ijk,
                          feat_stat=feat_stat)

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)

    def get_obj(self):
        raise NotImplementedError('invalid in base class Region, see subclass')

    @staticmethod
    def get_obj_pair(reg1, reg2, reg_union=None):
        """ this obj is to be minimized """
        # reg_union may be passed to reduce redundant computation
        if reg_union is None:
            reg_union = reg1 + reg2

        delta = reg_union.obj - (reg1.obj + reg2.obj)

        # flip if max value desired
        delta *= ((-1) ** reg1.max_flag)
        return delta


class RegionMinVar(Region):
    max_flag = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_grp = set(self.feat_stat.keys())

    def get_obj(self):
        var_sum = 0
        for grp in self.active_grp:
            fs = self.feat_stat[grp]
            var_sum = fs.cov_det * fs.n

        # assume uniform prior of grps
        return var_sum / len(self.active_grp)


class RegionMaxKL(Region):
    max_flag = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.active_grp is None:
            self.active_grp = list(self.feat_stat.keys())
            if len(self.active_grp) != 2:
                raise AttributeError('active_grp needed if > 2 grps')

    def get_obj(self):
        """ returns symmetric kullback liebler divergance * len(self.pc_ijk)

        (assumes each distribution is normal)

        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        kl(p_1, p_0) = .5 [log |sig_1| / |sig_0| -
                       d +
                       tr(sig_1 ^ -1 sig_0) + ...
                       (u_1 - u_0)^T sig_1 ^ -1 (u_1 - u_0)]

        NOTE: the first term cancels for symmetric kl:
        log |sig_1| / |sig_0| + log |sig_0| / |sig_1| = 0
        """
        grp_0, grp_1 = self.active_grp

        # init to d
        kl = len(self.feat_stat[grp_0].mu)

        # add other terms
        for fs_0, fs_1 in permutations((self.feat_stat[grp_0],
                                        self.feat_stat[grp_1])):
            kl += np.trace(fs_1.cov_inv @ fs_0.cov)
            mu_diff = fs_1.mu - fs_0.mu
            kl += mu_diff.T @ fs_1.cov_inv @ mu_diff

        return float(kl * len(self))


class RegionMaxMaha(Region):
    max_flag = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.active_grp is None:
            self.active_grp = list(self.feat_stat.keys())
            if len(self.active_grp) != 2:
                raise AttributeError('active_grp needed if > 2 grps')

    def get_obj(self):
        """ returns Mahalanobis between active_grp * len(self.pc_ijk)

        (assumes each distribution is normal and covar are equal).  Note that
        if covar between active_grp are equal then this is equivalent (and
        faster) than RegionMaxKL

        maha(p_1, p_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the common covariance
        """

        # compute common covariance
        fs_active = sum(self.feat_stat[grp] for grp in self.active_grp)

        # compute difference in mean
        grp_0, grp_1 = self.active_grp
        mu_diff = self.feat_stat[grp_0].mu - self.feat_stat[grp_1].mu

        # compute mahalanobis
        maha = mu_diff.T @ fs_active.cov_inv @ mu_diff

        return float(maha * len(self))
