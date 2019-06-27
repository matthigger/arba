import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(font_scale=1.2)


def append_col_one(x):
    if len(x.shape) == 1:
        x = np.atleast_2d(x).T
    num_sbj = x.shape[0]
    ones = np.atleast_2d(np.ones(num_sbj)).T
    return np.hstack((x, ones))


class RegionRegress:
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

    @staticmethod
    def from_data(pc_ijk, file_tree):
        assert file_tree.is_loaded, 'file tree must not loaded'
        mask = pc_ijk.to_mask()
        feat_img = np.mean(file_tree.data[mask, :, :], axis=0)
        return RegionRegress(feat_img=feat_img, pc_ijk=pc_ijk)

    @classmethod
    def set_feat_sbj(cls, feat_sbj):
        cls.feat_sbj = np.atleast_2d(feat_sbj)
        cls.feat_sbj = append_col_one(cls.feat_sbj)

        x = cls.feat_sbj
        cls.pseudo_inv = np.linalg.inv(x.T @ x) @ x.T

    def fit(self, feat_img):
        assert self.feat_sbj is not None, \
            'call RegRegress.set_feat_sbj() needed'

        self.beta = self.pseudo_inv @ feat_img

    def get_err(self, feat_img=None):
        if feat_img is None:
            feat_img = self.feat_img

        delta = feat_img - self.project(self.feat_sbj, append_flag=False)

        err_cov = np.atleast_2d(np.cov(delta.T, ddof=0))
        r2 = 1 - np.linalg.det(err_cov) / \
             np.linalg.det(self.feat_img_cov)

        return err_cov, r2

    def __init__(self, feat_img, pc_ijk):
        self.feat_img = np.atleast_2d(feat_img)
        self.feat_img_cov = np.atleast_2d(np.cov(self.feat_img.T, ddof=0))

        self.pc_ijk = pc_ijk

        self.fit(feat_img)
        self.err_cov, self.r2 = self.get_err(feat_img)

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
        delta = reg_u.r2 * len(reg_u) - \
                reg_1.r2 * len(reg_1) - \
                reg_2.r2 * len(reg_2)

        return np.abs(delta)

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
        plt.suptitle(f'r2={self.r2:.2f}, size={len(self)} vox')
        plt.plot(feat_sbj_line, feat_img_line)
        plt.xlabel(sbj_feat_label)
        plt.ylabel(img_feat_label)
