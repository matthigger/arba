from collections import defaultdict

import numpy as np

from .feat_stat import FeatStat
from .ward_grp import RegionWardGrp


class RegionWardSbj(RegionWardGrp):
    """ computes t2 according to split covariance model (across sbj and space)
    """

    @staticmethod
    def from_data(pc_ijk, file_tree, split, **kwargs):

        # get grp_sbj_fs_dict
        mask = pc_ijk.to_mask()
        grp_sbj_fs_dict = RegionWardSbj.get_grp_sbj_fs_dict(file_tree, mask,
                                                            split)

        # get stats across region per grp
        fs_dict = defaultdict(list)
        for ijk in pc_ijk:
            for grp, sbj_list in split.items():
                fs_dict[grp].append(file_tree.get_fs(ijk, sbj_list=sbj_list))
        fs_dict = {k: sum(l) for k, l in fs_dict.items()}

        return RegionWardSbj(pc_ijk=pc_ijk, fs_dict=fs_dict,
                             grp_sbj_fs_dict=grp_sbj_fs_dict, **kwargs)

    def __init__(self, *args, grp_sbj_fs_dict=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.grp_sbj_fs_dict = grp_sbj_fs_dict

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.fs_dict)

        if not isinstance(other, type(self)):
            raise TypeError

        fs_dict = dict()
        grp_sbj_fs_dict = dict()
        for grp in self.fs_dict.keys():
            fs_dict[grp] = self.fs_dict[grp] + \
                           other.fs_dict[grp]
            grp_sbj_fs_dict[grp] = self.grp_sbj_fs_dict[grp] + \
                                   other.grp_sbj_fs_dict[grp]

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict, grp_sbj_fs_dict=grp_sbj_fs_dict)

    @property
    def sig_sbj(self):
        return FeatStat.get_pool_cov(self.grp_sbj_fs_dict.values())

    @property
    def sig_space(self):
        return self.cov_pool - self.sig_sbj

    @property
    def n0n1(self):
        fs0, fs1 = self.grp_sbj_fs_dict.values()
        return fs0.n, fs1.n

    def get_t2(self):
        """ hotelling t squared distance between groups

        (n0 * n1) / (n0 + n1) (u_1 - u_0)^T sig^-1 (u_1 - u_0)

        where sig = sig_sbj + sig_space / len(self)
        """

        fs0, fs1 = self.fs_dict.values()
        mu_diff = fs0.mu - fs1.mu

        sig = self.sig_sbj + self.sig_space / len(self.pc_ijk)

        quad_term = mu_diff @ np.linalg.inv(sig) @ mu_diff

        n0, n1 = self.n0n1
        scale = (n0 * n1) / (n0 + n1)

        t2 = scale * quad_term

        if t2 < 0:
            raise AttributeError('invalid t2')

        return t2

    @staticmethod
    def get_grp_sbj_fs_dict(file_tree, mask, split):
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
        grp_sbj_fs_dict = dict()
        for grp, bool_idx in split.bool_iter():
            grp_sbj_fs_dict[grp] = FeatStat.from_array(x[bool_idx, :].T)

        return grp_sbj_fs_dict
