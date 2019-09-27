import numpy as np

import arba.space
from .reg import Region


class RegionDiscriminate(Region):
    """ merges regions to minimize population pooled covariance for discrim

    Class Attributes:
        grp_sbj_list_dict (dict): keys are groups, values are lists of sbj in
                                  the grp

    Instance Attributes:
        t2 (float): hotelling's t^2 (delta.T @ (cov_pool)^-1 @ delta) where
                    delta is difference in grp means and cov_pool is the grp
                    pooled covariance
    """
    grp_sbj_list_dict = None

    @classmethod
    def set_grp_sbj_list_dict(cls, grp_sbj_list_dict):
        cls.grp_sbj_list_dict = grp_sbj_list_dict

    @staticmethod
    def from_data_img(data_img, ijk=None, pc_ijk=None):
        assert RegionDiscriminate.grp_sbj_list_dict is not None, \
            'call set_grp_sbj_list_dict'

        fs_dict = dict()
        for grp, sbj_list in RegionDiscriminate.grp_sbj_list_dict.items():
            fs_dict[grp] = data_img.get_fs_dict(ijk=ijk, pc_ijk=pc_ijk)

        if pc_ijk is None:
            pc_ijk = arba.space.PointCloud([ijk], ref=data_img.ref)
        return RegionDiscriminate(pc_ijk=pc_ijk, fs_dict=fs_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delta, self.cov_pool, self.t2 = get_t2_stats(fs_dict=self.fs_dict)

        self.cov_pool_det = np.linalg.det(self.cov_pool)

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        if reg_u is None:
            reg_u = reg_1 + reg_2

        # min mean cov_pool_det of resultant segmentation
        err = reg_u.cov_pool_det * len(reg_u) - \
              reg_1.cov_pool_det * len(reg_1) - \
              reg_2.cov_pool_det * len(reg_2)

        assert err >= 0, 'err is negative'

        return err


def get_t2_stats(fs_dict, grp_tuple=None):
    if grp_tuple is None:
        grp0, grp1 = tuple(RegionDiscriminate.grp_sbj_list_dict.keys())
    else:
        grp0, grp1 = grp_tuple

    delta = fs_dict[grp1].mu - fs_dict[grp0].mu

    # get pool cov
    n0, n1 = fs_dict[grp0].n, fs_dict[grp1].n
    lam0, lam1 = n0 / (n0 + n1), n1 / (n0 + n1)
    cov_pool = fs_dict[grp0].cov * lam0 + \
               fs_dict[grp1].cov * lam1

    # compute t2
    t2 = delta @ np.linalg.inv(cov_pool) @ delta

    return delta, cov_pool, t2
