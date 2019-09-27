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

        # get delta (note: key order doesn't matter, -delta yields same t2)
        grp0, grp1 = tuple(RegionDiscriminate.grp_sbj_list_dict.keys())
        self.delta = self.fs_dict[grp1] - self.fs_dict[grp0]

        # get pool cov
        n0, n1 = self.fs_dict[grp0], self.fs_dict[grp1]
        lam0, lam1 = n0 / (n0 + n1), n1 / (n0 + n1)
        self.cov_pool = self.fs_dict[grp0].cov * lam0 + \
                        self.fs_dict[grp1].cov * lam1
        self.cov_pool_det = np.linalg.det(self.cov_pool)

        # compute t2
        self.t2 = self.delta @ np.linalg.inv(self.cov_pool) @ self.delta

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
