import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import arba.space
from .reg import Region

sns.set(font_scale=1.2)


def append_col_one(x):
    if len(x.shape) == 1:
        x = np.atleast_2d(x).T
    num_sbj = x.shape[0]
    ones = np.atleast_2d(np.ones(num_sbj)).T
    return np.hstack((x, ones))


class RegionRegress(Region):
    """ regression, from sbj features to img features, on some volume

    Class Attributes:
        feat_sbj (np.array): (num_sbj, dim_sbj + 1) subject features (ones col
                             appended)
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
    sbj_list = None

    @classmethod
    def set_feat_sbj(cls, feat_sbj, sbj_list):
        cls.sbj_list = sbj_list
        cls.feat_sbj = np.atleast_2d(feat_sbj)
        cls.feat_sbj = append_col_one(cls.feat_sbj)

        x = cls.feat_sbj
        cls.pseudo_inv = np.linalg.inv(x.T @ x) @ x.T

    def fit(self, feat_img):
        assert self.feat_sbj is not None, \
            'call RegRegress.set_feat_sbj() needed'

        self.beta = self.pseudo_inv @ feat_img

    def get_err(self):
        # covariance of imaging features around mean (per sbj)
        spatial_cov = sum(fs.cov for fs in self.fs_dict.values())
        delta = self.feat_img - self.project(self.feat_sbj, append_flag=False)
        regress_cov = delta.T @ delta
        err_cov = (spatial_cov + regress_cov) / len(self.sbj_list)

        cov = sum(self.fs_dict.values()).cov
        r2 = 1 - (np.trace(err_cov) / np.trace(cov))

        # compute log_like
        m = self.feat_img.shape[1]
        log_like = m * np.log(2 * np.pi) + np.log(np.linalg.det(err_cov))
        log_like *= len(self.sbj_list)
        c = np.linalg.inv(err_cov) @ (delta.T @ delta + spatial_cov)
        log_like += np.trace(c)
        log_like *= -.5 * len(self)

        return err_cov, r2, log_like

    @staticmethod
    def from_file_tree(file_tree, ijk=None, pc_ijk=None):
        assert (ijk is None) != (pc_ijk is None), 'ijk or pc_ijk required'
        if pc_ijk is None:
            pc_ijk = arba.space.PointCloud({ijk}, ref=file_tree.ref)

        fs_dict = {sbj: file_tree.get_fs(pc_ijk=pc_ijk, sbj_list=[sbj])
                   for sbj in RegionRegress.sbj_list}

        return RegionRegress(pc_ijk=pc_ijk, fs_dict=fs_dict)

    def __init__(self, pc_ijk, fs_dict):
        super().__init__(pc_ijk=pc_ijk, fs_dict=fs_dict)

        img_dim = next(iter(fs_dict.values())).d
        self.feat_img = np.ones(shape=(len(fs_dict), img_dim))
        for idx, sbj in enumerate(self.sbj_list):
            self.feat_img[idx, :] = self.fs_dict[sbj].mu

        self.feat_img_cov = np.atleast_2d(np.cov(self.feat_img.T, ddof=0))

        self.pc_ijk = pc_ijk
        self.beta = None
        self.fit(self.feat_img)
        self.err_cov, self.r2, self.log_like = self.get_err()
        self.mse = np.trace(self.err_cov)

    def __add__(self, other):
        # allows use of sum(reg_iter)
        if isinstance(other, type(0)) and other == 0:
            return type(self)(self.feat_img, self.pc_ijk)

        fs_dict = {sbj: self.fs_dict[sbj] + other.fs_dict[sbj]
                   for sbj in self.sbj_list}

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict)

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        if reg_u is None:
            reg_u = reg_1 + reg_2

        err = reg_u.mse * len(reg_u) - \
              reg_1.mse * len(reg_1) - \
              reg_2.mse * len(reg_2)

        assert err >= 0, 'err is negative'

        return err

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)

    def project(self, feat_sbj, append_flag=True):
        if append_flag:
            feat_sbj = append_col_one(feat_sbj)
        return feat_sbj @ self.beta

    def plot(self, sbj_feat_label='sbj_feat', img_feat_label='img_feat'):
        # line below note: we append column of 1's ...
        assert self.feat_sbj.shape[1] == 2, 'only valid for scalar feat_sbj'
        assert self.feat_img.shape[1] == 1, 'only valid for scalar feat_img'

        feat_sbj = self.feat_sbj[:, 0]
        feat_sbj_line = np.array((min(feat_sbj), max(feat_sbj)))
        feat_img_line = self.project(feat_sbj_line, append_flag=True)

        plt.scatter(feat_sbj, self.feat_img)
        plt.suptitle(
            f'mse={self.mse:.2f}, r2={self.r2:.2f}, size={len(self)} vox')
        plt.plot(feat_sbj_line, feat_img_line)
        plt.xlabel(sbj_feat_label)
        plt.ylabel(img_feat_label)
