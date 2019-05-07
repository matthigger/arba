import numpy as np

from .feat_stat import FeatStat
from .pool_cov import PoolCov
from .ward_grp import RegionWardGrp


class RegionWardSbj(RegionWardGrp):
    """ computes t2 according to split covariance model (across sbj and space)
    """

    @classmethod
    def from_data(cls, pc_ijk, file_tree, split, **kwargs):

        reg = super().from_data(pc_ijk, file_tree, split, **kwargs)

        mask = pc_ijk.to_mask()
        reg.sig_sbj = RegionWardSbj.get_sig_sbj(file_tree, mask, split)
        reg.sig_space = PoolCov(n=reg.sig_sbj.n,
                                value=reg.cov_pool.value -
                                      reg.sig_sbj.value)

        return reg

    def __init__(self, *args, sig_sbj=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.sig_sbj = sig_sbj
        if sig_sbj is not None:
            self.sig_space = PoolCov(n=self.sig_sbj.n,
                                     value=self.cov_pool.value - sig_sbj.value)

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.fs_dict)

        if not isinstance(other, type(self)):
            raise TypeError

        fs_dict = {grp: self.fs_dict[grp] + other.fs_dict[grp]
                   for grp in self.fs_dict.keys()}
        sig_sbj = PoolCov.sum((self.sig_sbj, other.sig_sbj))

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict, sig_sbj=sig_sbj)

    @property
    def n0n1(self):
        fs0, fs1 = self.fs_dict.values()
        return fs0.n / len(self.pc_ijk), fs1.n / len(self.pc_ijk)

    def get_t2(self):
        """ hotelling t squared distance between groups

        (n0 * n1) / (n0 + n1) (u_1 - u_0)^T sig^-1 (u_1 - u_0)

        where sig = sig_sbj + sig_space / len(self)
        """

        fs0, fs1 = self.fs_dict.values()
        mu_diff = fs0.mu - fs1.mu

        sig = self.sig_sbj.value + self.sig_space.value / len(self.pc_ijk)

        quad_term = mu_diff @ np.linalg.inv(sig) @ mu_diff

        n0, n1 = self.n0n1
        scale = (n0 * n1) / (n0 + n1)

        t2 = scale * quad_term

        if t2 < 0:
            raise AttributeError('invalid t2')

        return t2

    @staticmethod
    def get_sig_sbj(file_tree, mask, split):
        """ computes sig_sbj from raw data

        Args:
            file_tree (FileTree): data object
            mask (np.array): boolean array, describes space of region
            split (Split):
        """
        # get relevant features
        with file_tree.loaded():
            x = file_tree.data[mask, :, :]

        # average per sbj
        x = np.mean(x, axis=0)

        # compute covariance
        sig_sbj_list = list()
        for grp, bool_idx in split.bool_iter():
            fs = FeatStat.from_array(x[bool_idx, :].T)
            sig_sbj_list.append(PoolCov.from_feat_stat(fs))

        return PoolCov.sum(sig_sbj_list)
