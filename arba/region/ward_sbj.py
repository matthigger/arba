from collections import defaultdict

import numpy as np

from .ward_grp import RegionWardGrp


class RegionWardSbj(RegionWardGrp):
    """ computes t2 according to split covariance model (across sbj and space)
    """

    @staticmethod
    def from_data(pc_ijk, file_tree, grp_sbj_dict, **kwargs):

        # get sig_sbj_dict
        mask = pc_ijk.to_mask()
        sig_sbj_dict = RegionWardSbj.get_sig_sbj(file_tree, mask, grp_sbj_dict)

        # get stats across region per grp
        fs_dict = defaultdict(list)
        for ijk in pc_ijk:
            for grp, sbj_list in grp_sbj_dict.items():
                fs_dict[grp].append(file_tree.get_fs(ijk, sbj_list=sbj_list))
        fs_dict = {k: sum(l) for k, l in fs_dict.items()}

        return RegionWardSbj(pc_ijk=pc_ijk, fs_dict=fs_dict,
                             sig_sbj_dict=sig_sbj_dict, **kwargs)

    def __init__(self, *args, sig_sbj_dict, **kwargs):

        super().__init__(*args, **kwargs)

        self.sig_sbj_dict = sig_sbj_dict

        self.sig_space_dict = dict()
        for grp in self.sig_sbj_dict.keys():
            self.sig_space_dict[grp] = self.fs_dict[grp].cov - \
                                       sig_sbj_dict[grp]

            assert (self.sig_space_dict[
                        grp] >= 0).all(), 'invalid sig_sbj passed'

    @property
    def n0n1(self):
        fs0, fs1 = self.fs_dict.values()
        # assume same num of sbj per each vox
        num_vox = len(self.pc_ijk)
        return fs0.n / num_vox, fs1.n / num_vox

    def get_t2(self):
        """ hotelling t squared distance between groups

        (n0 * n1) / (n0 + n1) (u_1 - u_0)^T sig^-1 (u_1 - u_0)

        where sig = sig_sbj + sig_space / len(self)
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu
        sig_list = list()
        for grp in self.sig_space_dict.keys():
            sig = self.sig_sbj_dict[grp] + self.sig_space_dict[grp] / len(
                self.pc_ijk)
            sig_list.append(sig)

        # pool
        sig = np.atleast_2d(np.mean(sig_list))

        quad_term = mu_diff @ np.linalg.inv(sig) @ mu_diff

        n0, n1 = self.n0n1
        scale = (n0 * n1) / (n0 + n1)

        t2 = scale * quad_term

        if t2 < 0:
            raise AttributeError('invalid t2')

        return t2

    @staticmethod
    def get_sig_sbj(file_tree, mask, grp_sbj_dict):
        """ computes sig_sbj from raw data

        Args:
            file_tree (FileTree): data object
            mask (np.array): boolean array, describes space of region
            grp_sbj_dict (dict): keys are group labels, values are list of sbj
        """
        # get relevant features
        with file_tree.loaded():
            x = file_tree.data[mask, :, :]

        # average per sbj
        x = np.mean(x, axis=0)

        # compute covariance
        sig_sbj_dict = dict()
        for grp, sbj_list in grp_sbj_dict.items():
            split = np.array([sbj in sbj_list for sbj in file_tree.sbj_list])
            sig_sbj = np.cov(x[split, :].T, ddof=0)
            sig_sbj_dict[grp] = np.atleast_2d(sig_sbj)

        return sig_sbj_dict
