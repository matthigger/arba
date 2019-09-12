import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import arba.space
from .feat_stat import FeatStatSingle
from .reg import Region
from ..effect import compute_r2


class RegionRegress(Region):
    """ regression, from sbj features to img features, on some volume

    Class Attributes:
        data_sbj (DataSubject):

    Instance Attributes:
        feat_img (np.array): (num_sbj, dim_img) mean features observed (per
                              sbj) across the entire region
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        beta (np.array): (dim_sbj, dim_img) mapping from img to sbj space
        err_cov (np.array): (dim_img, dim_img) error covariance observed
        err_cov_det (float): covariance of err_cov
    """
    data_sbj = None

    @classmethod
    def set_data_sbj(cls, data_sbj):
        cls.data_sbj = data_sbj
        assert np.all(data_sbj.contrast), 'nuisance params not supported'

    @staticmethod
    def from_data_img(data_img, ijk=None, pc_ijk=None):
        fs_dict = RegionRegress.get_fs_dict(data_img, ijk=ijk, pc_ijk=pc_ijk)
        if pc_ijk is None:
            pc_ijk = arba.space.PointCloud([ijk], ref=data_img.ref)
        return RegionRegress(pc_ijk=pc_ijk, fs_dict=fs_dict)

    @staticmethod
    def get_fs_dict(data_img, ijk=None, pc_ijk=None):
        assert data_img.is_loaded, 'file tree must be loaded'
        assert (ijk is None) != (pc_ijk is None), 'ijk or pc_ijk required'

        if pc_ijk is None:
            # slight computational speedup for a single voxel
            i, j, k = ijk
            fs_dict = dict()
            for sbj_idx, sbj in enumerate(data_img.sbj_list):
                x = data_img.data[i, j, k, sbj_idx, :]
                fs_dict[sbj] = FeatStatSingle(x)
        else:
            # computationally slower
            fs_dict = {sbj: data_img.get_fs(pc_ijk=pc_ijk, sbj_list=[sbj])
                       for sbj in data_img.sbj_list}

        return fs_dict

    def __init__(self, pc_ijk, fs_dict, _beta=None):
        super().__init__(pc_ijk=pc_ijk, fs_dict=fs_dict)
        assert self.data_sbj is not None, 'set_data_sbj() not called'

        img_dim = next(iter(fs_dict.values())).d
        self.feat_img = np.empty(shape=(len(fs_dict), img_dim))
        for idx, sbj in enumerate(self.data_sbj.sbj_list):
            self.feat_img[idx, :] = self.fs_dict[sbj].mu

        # fit beta
        self.beta = _beta
        if self.beta is None:
            self.beta = self.data_sbj.pseudo_inv @ self.feat_img

        # covariance of imaging features around sbj mean
        self.space_cov_pool = sum(fs.cov for fs in self.fs_dict.values()) / \
                              self.data_sbj.num_sbj

        # r2
        self.r2 = compute_r2(x=self.data_sbj.feat,
                             y=self.feat_img,
                             beta=self.beta,
                             y_pool_cov=self.space_cov_pool)

    def __add__(self, other):
        # allows use of sum(reg_iter)
        if isinstance(other, type(0)) and other == 0:
            return type(self)(self.feat_img, self.pc_ijk)

        fs_dict = {sbj: self.fs_dict[sbj] + other.fs_dict[sbj]
                   for sbj in self.data_sbj.sbj_list}

        lambda_self = len(self) / (len(self) + len(other))
        beta = lambda_self * self.beta + (1 - lambda_self) * other.beta

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict,
                          _beta=beta)

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

    def plot(self, img_idx, sbj_idx, img_label='image feat',
             sbj_label='sbj feat'):
        sns.set()
        x = self.data_sbj.feat[:, sbj_idx]
        y = self.feat_img[:, img_idx]

        feat_sbj_line = self.data_sbj.feat.mean(axis=0)
        feat_sbj_line = np.repeat(np.atleast_2d(feat_sbj_line), repeats=2,
                                  axis=0)
        feat_sbj_line[:, sbj_idx] = min(x), max(x)
        feat_img_line = feat_sbj_line @ self.beta

        plt.scatter(x, y, label='single sbj (region mean)')
        plt.suptitle(', '.join([f'r2_vox={self.r2:.2f}',
                                f'size={len(self)} vox']))
        plt.plot(feat_sbj_line[:, sbj_idx], feat_img_line, label='beta')
        plt.xlabel(sbj_label)
        plt.ylabel(img_label)
        plt.legend()
