from itertools import permutations
import numpy as np
from .feat_stat import FeatStatSingle
from .reg import Region


class RegionKL(Region):
    @property
    def kl(self):
        return self.obj

    @property
    def error(self):
        """ variance of kl distance (per voxel)

        there is a kl distance per voxel, as voxels are aggregated the kl
        of the entire region stands in.  this is a measurement of how accurate
        the kl of the region represents the kl of its constituent voxels
        """
        return float(self.error_fs.cov) * len(self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj = None
        self.error_fs = FeatStatSingle(mu=self.kl)

    def get_obj(self):
        """ negative symmetric kullback liebler divergance * len(self.pc_ijk)

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
        kl = self.fs_dict[grp_0].d

        # add other terms
        for fs_0, fs_1 in permutations((self.fs_dict[grp_0],
                                        self.fs_dict[grp_1])):
            kl += np.trace(fs_1.cov_inv @ fs_0.cov)
            mu_diff = fs_1.mu - fs_0.mu
            kl += mu_diff @ fs_1.cov_inv @ mu_diff

        return kl

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            reg_out = super().__add__(other)
            reg_out.error_fs = self.error_fs
            return reg_out

        if not isinstance(other, type(self)):
            raise TypeError

        reg_out = super().__add__(other)
        reg_out.error_fs = self.error_fs + other.error_fs

        if reg_out.error_fs.n != len(reg_out):
            raise AttributeError

        return reg_out

    __radd__ = __add__
