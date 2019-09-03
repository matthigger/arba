import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import arba.space
from .feat_stat import FeatStatSingle
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
        feat_sbj (np.array): (num_sbj, dim_sbj) subject features
        pseudo_inv (np.array): pseudo inverse (used to compute beta)

    Instance Attributes:
        feat_img (np.array): (num_sbj, dim_img) mean features observed (per
                              sbj) across the entire region
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        beta (np.array): (dim_sbj, dim_img) mapping from img to sbj space
        err_cov (np.array): (dim_img, dim_img) error covariance observed
        err_cov_det (float): covariance of err_cov
    """

    feat_sbj = None
    pseudo_inv = None
    sbj_list = None
    num_sbj = None

    @property
    def dim_sbj(self):
        return self.feat_sbj.shape[1]

    @classmethod
    def set_feat_sbj(cls, feat_sbj, sbj_list, append_ones=True):
        cls.sbj_list = sbj_list
        cls.num_sbj = len(sbj_list)
        cls.feat_sbj = np.atleast_2d(feat_sbj)
        if append_ones:
            cls.feat_sbj = append_col_one(cls.feat_sbj)

        x = cls.feat_sbj

        # build pseudo inv (store intermediate states for later computing)
        cls.x_trans_x = x.T @ x
        cls.x_trans_x_inv = np.linalg.inv(cls.x_trans_x)
        cls.pseudo_inv = cls.x_trans_x_inv @ x.T

    @classmethod
    def shuffle_feat_sbj(cls, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.array(range(cls.feat_sbj.shape[0]))
        np.random.shuffle(idx)
        cls.set_feat_sbj(feat_sbj=cls.feat_sbj[idx, :],
                         sbj_list=cls.sbj_list,
                         append_ones=False)

    def fit(self, feat_img):
        assert self.feat_sbj is not None, \
            'call RegRegress.set_feat_sbj() needed'

        self.beta = self.pseudo_inv @ feat_img

    @staticmethod
    def from_file_tree(file_tree, ijk=None, pc_ijk=None):
        assert (ijk is None) != (pc_ijk is None), 'ijk or pc_ijk required'
        assert RegionRegress.sbj_list == file_tree.sbj_list

        if pc_ijk is None:
            # slight computational speedup for a single voxel
            pc_ijk = arba.space.PointCloud([ijk], ref=file_tree.ref)
            i, j, k = ijk
            fs_dict = dict()
            for sbj_idx, sbj in enumerate(RegionRegress.sbj_list):
                x = file_tree.data[i, j, k, sbj_idx, :]
                fs_dict[sbj] = FeatStatSingle(x)
        else:
            # computationally slower
            fs_dict = {sbj: file_tree.get_fs(pc_ijk=pc_ijk, sbj_list=[sbj])
                       for sbj in RegionRegress.sbj_list}

        return RegionRegress(pc_ijk=pc_ijk, fs_dict=fs_dict)

    def __init__(self, pc_ijk, fs_dict, beta=None):
        super().__init__(pc_ijk=pc_ijk, fs_dict=fs_dict)

        img_dim = next(iter(fs_dict.values())).d
        self.feat_img = np.empty(shape=(len(fs_dict), img_dim))
        for idx, sbj in enumerate(self.sbj_list):
            self.feat_img[idx, :] = self.fs_dict[sbj].mu

        # fit beta
        self.beta = beta
        if self.beta is None:
            self.fit(self.feat_img)

        # covariance of imaging features around mean (per sbj)
        self.space_cov_pool = sum(fs.cov for fs in self.fs_dict.values()) / \
                              self.num_sbj
        delta = self.feat_img - self.project(self.feat_sbj, append_flag=False)
        self.eps_mean = delta.T @ delta / self.num_sbj

        self.eps = self.space_cov_pool + self.eps_mean

        # compute derivative stats
        self.mse = np.trace(self.eps)
        self.mse_mean = np.trace(self.eps_mean)

        self.cov = sum(self.fs_dict.values()).cov
        self.cov_mean = self.cov - self.space_cov_pool

        self.r2 = 1 - (np.trace(self.eps) / np.trace(self.cov))
        self.r2_mean = 1 - (np.trace(self.eps_mean) / np.trace(self.cov_mean))

        # compute mahalanobis to each sbj fature
        n = self.num_sbj * len(self)
        self.maha = np.empty(self.dim_sbj)
        for idx, eps in enumerate(np.diag(self.eps)):
            s = eps * n / (n - 1) * self.x_trans_x_inv
            self.maha[idx] = (self.beta.T @ s @ self.beta)[0, 0]

    def __add__(self, other):
        # allows use of sum(reg_iter)
        if isinstance(other, type(0)) and other == 0:
            return type(self)(self.feat_img, self.pc_ijk)

        fs_dict = {sbj: self.fs_dict[sbj] + other.fs_dict[sbj]
                   for sbj in self.sbj_list}

        lambda_self = len(self) / (len(self) + len(other))
        beta = lambda_self * self.beta + (1 - lambda_self) * other.beta

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict,
                          beta=beta)

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        if reg_u is None:
            reg_u = reg_1 + reg_2

        # min mean squared error of resultant segmentation
        # err = reg_u.mse * len(reg_u) - \
        #       reg_1.mse * len(reg_1) - \
        #       reg_2.mse * len(reg_2)

        # max mean r2 of resultant segmentation
        err = -(reg_u.r2 * len(reg_u) -
                reg_1.r2 * len(reg_1) -
                reg_2.r2 * len(reg_2))

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
        plt.suptitle(', '.join([f'mse={self.mse:.2f}',
                                f'r2_vox={self.r2:.2f}',
                                f'r2_mean={self.r2_mean: .2f}',
                                f'size={len(self)} vox']))
        plt.plot(feat_sbj_line, feat_img_line)
        plt.xlabel(sbj_feat_label)
        plt.ylabel(img_feat_label)
