import numpy as np


class RegionRegress:
    """ regression, from sbj features to img features, on some volume

    Class Attributes:
        feat_sbj (np.array): (num_sbj, dim_sbj) subject features
        pseudo_inv (np.array): pseudo inverse (used to compute beta)

    Instance Attributes:
        feat_img (np.array): (num_sbj, dim_img) mean features observed (per
                              sbj) across the entire region
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        beta (np.array): (dim_sbj + 1, dim_img) mapping from img to sbj space
        err_cov (np.array): (dim_img, dim_img) error covariance observed
        err_cov_det (float): covariance of err_cov
    """

    feat_sbj = None
    pseudo_inv = None

    @staticmethod
    def from_data(pc_ijk, file_tree):
        mask = pc_ijk.to_mask()
        feat_img = np.mean(file_tree.data[mask, :, :], axis=0)
        return RegionRegress(feat_img=feat_img, pc_ijk=pc_ijk)

    @classmethod
    def set_feat_sbj(cls, feat_sbj):
        cls.feat_sbj = feat_sbj

        cls.pseudo_inv = np.linalg.inv(feat_sbj.T @ feat_sbj) @ feat_sbj.T

    def _compute_beta_err_cov(self):
        assert self.feat_sbj is not None, \
            'call RegRegress.set_feat_sbj() needed'

        self.beta = self.pseudo_inv @ self.feat_img
        estimate = self.feat_img @ self.beta.T
        delta = self.feat_sbj - estimate
        self.err_cov = np.cov(delta.T, ddof=0)
        self.err_cov_det = np.linalg.det(self.err_cov)

    def __init__(self, feat_img, pc_ijk):
        self.feat_img = feat_img
        self.pc_ijk = pc_ijk

        self._compute_beta_err_cov()

    def __add__(self, other):
        # allows use of sum(reg_iter)
        if isinstance(other, type(0)) and other == 0:
            return type(self)(self.feat_img, self.pc_ijk)

        # compute feat_img
        lam = len(self) / (len(self) + len(other))
        feat_img = self.feat_img * lam + \
                   other.feat_img * (1 - lam)

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          feat_img=feat_img)

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        if reg_u is None:
            reg_u = reg_1 + reg_2
        delta = reg_u.err_cov_det * len(reg_u) - \
                reg_1.err_cov_det * len(reg_1) - \
                reg_2.err_cov_det * len(reg_2)
        return delta

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)
