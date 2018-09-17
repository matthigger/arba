from itertools import permutations

import numpy as np


class Region:
    """ a set of voxels and their associated features

    Attributes:
        pc_ijk (PointCloudIJK)
        feat_stat (dict): contains feat stats for img sets of different grps
    """

    @property
    def obj(self):
        if self._obj is None:
            self._obj = self.get_obj()
        return self._obj

    def __init__(self, pc_ijk, feat_stat):
        self.pc_ijk = pc_ijk
        self.feat_stat = feat_stat
        self._obj = None

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
    def get_obj_pair(reg1, reg2):
        return (reg1 + reg2).obj - (reg1.obj + reg2.obj)


class RegionMinVar(Region):
    def __init__(self, *args, grp_to_min_var=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.grp_to_min_var = grp_to_min_var
        if self.grp_to_min_var is None:
            self.grp_to_min_var = set(self.feat_stat.keys())

    def get_obj(self):
        var_sum = 0
        for grp in self.grp_to_min_var:
            fs = self.feat_stat[grp]
            var_sum = np.linalg.det(fs.var) * fs.n

        # assume uniform prior of grps
        return var_sum / len(self.grp_to_min_var)


class RegionKL(Region):
    def __init__(self, *args, grp_to_max_kl=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.grp_to_max_kl = grp_to_max_kl
        if self.grp_to_max_kl is None:
            self.grp_to_max_kl = list(self.feat_stat.keys())
            if len(self.grp_to_max_kl) != 2:
                raise AttributeError('grp_to_max_kl needed if > 2 grps')

    def get_obj(self):
        """ returns symmetric kullback liebler divergance (assumes normal)

        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        kl(p_1, p_0) = .5 [log |sig_1| / |sig_0| -
                       d +
                       tr(sig_1 ^ -1 sig_0) + ...
                       (u_1 - u_0)^T sig_1 ^ -1 (u_1 - u_0)]

        note: the first term cancels for symmetric kl:
        log |sig_1| / |sig_0| + log |sig_0| / |sig_1| = 0
        """
        grp_0, grp_1 = self.grp_to_max_kl

        kl = len(self.feat_stat[grp_0].mu)
        for fs_0, fs_1 in permutations((self.feat_stat[grp_0],
                                       self.feat_stat[grp_1])):
            fs_1_var_inv = np.linalg.inv(fs_1.var)
            kl += np.trace(fs_1_var_inv @ fs_0.var)
            mu_diff = fs_1.mu - fs_0.mu
            kl += mu_diff.T @ fs_1_var_inv @ mu_diff

        return float(kl * len(self))
