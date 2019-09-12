import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import arba.space
from .feat_stat import FeatStatSingle
from .reg import Region
from ..effect import compute_r2

sns.set(font_scale=1.2)


class RegionRegress(Region):
    """ regression, from sbj features to img features, on some volume

    Class Attributes:
        sbj_feat (SubjectFeatures):

    Instance Attributes:
        feat_img (np.array): (num_sbj, dim_img) mean features observed (per
                              sbj) across the entire region
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        beta (np.array): (dim_sbj, dim_img) mapping from img to sbj space
        err_cov (np.array): (dim_img, dim_img) error covariance observed
        err_cov_det (float): covariance of err_cov
    """
    sbj_feat = None

    @classmethod
    def set_sbj_feat(cls, sbj_feat):
        cls.sbj_feat = sbj_feat
        assert np.all(sbj_feat.contrast), 'nuisance params not supported'

    @staticmethod
    def from_file_tree(file_tree, ijk=None, pc_ijk=None):
        fs_dict = RegionRegress.get_fs_dict(file_tree, ijk=ijk, pc_ijk=pc_ijk)
        if pc_ijk is None:
            pc_ijk = arba.space.PointCloud([ijk], ref=file_tree.ref)
        return RegionRegress(pc_ijk=pc_ijk, fs_dict=fs_dict)

    @staticmethod
    def get_fs_dict(file_tree, ijk=None, pc_ijk=None):
        assert file_tree.is_loaded, 'file tree must be loaded'
        assert (ijk is None) != (pc_ijk is None), 'ijk or pc_ijk required'

        if pc_ijk is None:
            # slight computational speedup for a single voxel
            i, j, k = ijk
            fs_dict = dict()
            for sbj_idx, sbj in enumerate(file_tree.sbj_list):
                x = file_tree.data[i, j, k, sbj_idx, :]
                fs_dict[sbj] = FeatStatSingle(x)
        else:
            # computationally slower
            fs_dict = {sbj: file_tree.get_fs(pc_ijk=pc_ijk, sbj_list=[sbj])
                       for sbj in file_tree.sbj_list}

        return fs_dict

    def __init__(self, pc_ijk, fs_dict, _beta=None):
        super().__init__(pc_ijk=pc_ijk, fs_dict=fs_dict)
        assert self.sbj_feat is not None, 'set_sbj_feat() not called'

        img_dim = next(iter(fs_dict.values())).d
        self.feat_img = np.empty(shape=(len(fs_dict), img_dim))
        for idx, sbj in enumerate(self.sbj_feat.sbj_list):
            self.feat_img[idx, :] = self.fs_dict[sbj].mu

        # fit beta
        self.beta = _beta
        if self.beta is None:
            self.beta = self.sbj_feat.pseudo_inv @ self.feat_img

        # covariance of imaging features around sbj mean
        self.space_cov_pool = sum(fs.cov for fs in self.fs_dict.values()) / \
                              self.sbj_feat.num_sbj

        # r2
        self.r2 = compute_r2(x=self.sbj_feat.x,
                             y=self.feat_img,
                             beta=self.beta,
                             y_pool_cov=self.space_cov_pool)

    def __add__(self, other):
        # allows use of sum(reg_iter)
        if isinstance(other, type(0)) and other == 0:
            return type(self)(self.feat_img, self.pc_ijk)

        fs_dict = {sbj: self.fs_dict[sbj] + other.fs_dict[sbj]
                   for sbj in self.sbj_feat.sbj_list}

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

    def plot(self, sbj_feat_label='sbj_feat', img_feat_label='img_feat'):
        raise NotImplementedError
        # todo: assert self.feat_sbj.shape[1] == 1, 'only valid for scalar feat_sbj'
        assert self.feat_img.shape[1] == 1, 'only valid for scalar feat_img'

        # todo: feat_sbj = self.feat_sbj[:, 0]
        feat_sbj_line = np.array((min(feat_sbj), max(feat_sbj)))
        feat_img_line = feat_sbj_line @ self.beta

        plt.scatter(feat_sbj, self.feat_img)
        plt.suptitle(', '.join([f'mse={self.mse:.2f}',
                                f'r2_vox={self.r2:.2f}',
                                f'r2_mean={self.r2_mean: .2f}',
                                f'size={len(self)} vox']))
        plt.plot(feat_sbj_line, feat_img_line)
        plt.xlabel(sbj_feat_label)
        plt.ylabel(img_feat_label)
