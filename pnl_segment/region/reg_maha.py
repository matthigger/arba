from .feat_stat import get_maha_from_fs
from .reg_kl import RegionKL


class RegionMaha(RegionKL):
    @property
    def maha(self):
        return self._obj

    def get_obj(self, active_grp=None):
        """ negative Mahalanobis squared between active_grp

        (assumes each distribution is normal and covar are equal).  Note that
        if covar between active_grp are equal then this is equivalent (and
        faster) than RegionMaxKL

        maha(p_1, p_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the common covariance / region size (in voxels)
        """

        if active_grp is None:
            active_grp = self.active_grp

        fs0, fs1 = (self.fs_dict[grp] for grp in active_grp)

        return get_maha_from_fs(fs0, fs1)
