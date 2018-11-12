import os
import tempfile
from copy import deepcopy
from functools import reduce

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import mannwhitneyu

from ..region import FeatStat
from ..space import Mask


class Effect:
    """ adds an effect to an image

    note: only iid normal effects are supported

    Attributes:
        mean (np.array): average offset of effect on a voxel
        cov (np.array): square matrix, noise power
        mask (Mask): effect location
    """

    @staticmethod
    def from_data(ijk_fs_dict, mask, snr, cov_ratio=0, u=None):
        """ scales effect with observations

        Args:
            ijk_fs_dict (dict): keys are ijk, values are feat_stat (needed to
                                compute background variance)
            mask (Mask): effect location
            snr (float): ratio of effect to population variance
            cov_ratio (float): ratio of effect cov to population cov
            u (array): direction of offset
        """

        if snr < 0:
            raise AttributeError('snr must be positive')

        # get feat stat across mask
        fs = sum(ijk_fs_dict[ijk] for ijk in mask.to_point_cloud())

        # get direction u
        if u is None:
            u = np.diag(fs.cov)
        elif len(u) != fs.d:
            raise AttributeError('direction offset must have same len as fs.d')

        # compute mean offset which yields snr
        c = u @ fs.cov_inv @ u
        mean = u * np.sqrt(snr / c)

        cov = fs.cov * cov_ratio

        return Effect(mask, mean=mean, cov=cov, snr=snr)

    def __init__(self, mask, mean, cov=None, snr=None):
        if not isinstance(mask, Mask):
            raise TypeError(f'mask: {mask} must be of type Mask')
        self.mask = mask
        self.mean = np.reshape(mean, len(mean))
        if len(self.mean) ** 2 != len(cov.flatten()):
            raise AttributeError('mean / cov mismatch')
        self.cov = np.atleast_2d(cov)
        self.snr = snr

    def __len__(self):
        return len(self.mask)

    def apply_from_to_nii(self, f_nii_dict, f_nii_dict_out=None):
        def load(f_nii):
            """ loads f_nii, ensures proper space
            """
            img = nib.load(str(f_nii))
            if self.mask.ref is not None and \
                    not np.array_equal(img.affine, self.mask.ref.affine):
                raise AttributeError('space mismatch')
            return img.get_data()

        # init to tempfile
        if f_nii_dict_out is None:
            f_nii_dict_out = dict()
            for feat in f_nii_dict.keys():
                _, f_nii_dict_out[feat] = tempfile.mkstemp(suffix='.nii.gz')

        # load and stack data
        x_list = [load(f_nii_dict[label]) for label in self.feat_label]
        x = np.stack(x_list, axis=len(x_list[0].shape))

        # apply effect
        x = self.apply(x)

        # output img to nii
        for feat_idx in range(x.shape[-1]):
            feat = self.feat_label[feat_idx]
            affine = nib.load(str(f_nii_dict[feat])).affine
            img = nib.Nifti1Image(x[..., feat_idx], affine)
            img.to_filename(str(f_nii_dict_out[feat]))

        return f_nii_dict_out

    def get_auc(self, x, mask):
        """ computes auc of statistic given by array x

        a strong auc value requires that there exists some threshold which
        seperates affected voxels from unaffected voxels.  here, self serves
        as 'ground truth' of which voxels are, or are not, affected

        Args:
            x (np.array)
            mask (mask): values in x which are to be counted

        Returns:
            auc (float): value in [0, 1]
        """
        # mask the statistic
        stat_vec = x[mask]

        # mask the ground truth to relevant area
        truth_vec = self.mask[mask]

        return Effect._get_auc(stat_vec, truth_vec)

    def get_auc_from_nii(self, f_nii):
        """ computes auc of statistic given in f_nii

        Args:
            f_nii (str or Path): path to statistic image

        Returns:
            auc (float): value in [0, 1]
        """
        stat_mask = Mask.from_nii(f_nii)
        stat_vec = stat_mask.apply_from_nii(f_nii)
        truth_vec = stat_mask.apply(self.mask.x)

        return Effect.__get_auc(stat_vec, truth_vec)

    @staticmethod
    def _get_auc(stat_vec, truth_vec):

        x = stat_vec[truth_vec == 0]
        y = stat_vec[truth_vec == 1]
        try:
            u = mannwhitneyu(x, y, alternative='greater')
        except ValueError:
            # all values are same
            return .5
        auc = u.statistic / (len(x) * len(y))
        auc = max(auc, 1 - auc)
        # pval = min(u.pvalue, 1 - u.pvalue)

        return auc

    def apply(self, x):
        """ applies effect to array x

        Args:
            x (np.array): img to apply effect to
        """

        # sample effect
        effect = np.random.multivariate_normal(mean=self.mean,
                                               cov=self.cov,
                                               size=len(self))

        # add effect
        raise NotImplementedError
        x = self.mask.insert(x, effect, add=True)

        return x

    def apply_to_file_tree(self, file_tree, copy=True):
        if any(self.cov.flatten()):
            raise AttributeError('effect cov must be 0')

        if copy:
            file_tree = deepcopy(file_tree)

        ijk_set = self.mask.to_point_cloud()
        ijk_set &= {x for x in file_tree.ijk_fs_dict.keys()}
        for ijk in ijk_set:
            fs = file_tree.ijk_fs_dict[ijk]

            file_tree.ijk_fs_dict[ijk] = FeatStat(n=fs.n,
                                                  mu=fs.mu + self.mean,
                                                  cov=fs.cov)

        return file_tree

    @staticmethod
    def sample_mask(prior_array, radius, n=1, seg_array=None):
        """n effect centers chosen and dilated to build mask of affected volume

        Args:
            prior_array (np.array): has same shape as image.  values are
                                    unweighted prob of effect center.
                                    note that prior array also masks effect (no
                                    effect exists where prior_array <= 0)
            radius (int): effect radius, center is dilated by this amount
            n (int): number of effects
            seg_array (np.array): segmentation array, if passed constrains
                                      effect to the region of its center. has
                                      same shape as prior_array

        Returns:
            eff_mask (np.array): array of 1s where effect is present

        >>> np.random.seed(1)
        >>> prior_array = np.ones((10, 10))
        >>> m = Effect.sample_mask(prior_array, radius=3)
        >>> m.x
        array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        >>> seg_array = sum(Effect.sample_mask(prior_array, radius=5)
        ...                 for _ in range(3)).x
        >>> seg_array
        array([[2, 2, 2, 1, 1, 1, 0, 0, 0, 0],
               [2, 2, 2, 2, 1, 0, 0, 0, 0, 0],
               [2, 2, 3, 2, 1, 0, 0, 0, 0, 0],
               [2, 3, 3, 2, 1, 1, 0, 0, 0, 0],
               [3, 3, 2, 2, 2, 0, 0, 0, 0, 0],
               [3, 2, 2, 2, 1, 1, 0, 0, 0, 0],
               [2, 2, 2, 1, 1, 1, 1, 0, 0, 0],
               [2, 2, 1, 1, 1, 1, 1, 1, 0, 0],
               [2, 1, 1, 1, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        >>> Effect.sample_mask(prior_array, radius=7, seg_array=seg_array).x
        array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        """

        shape = prior_array.shape

        # choose effect centers via prior array (idx is linear idx)
        p = prior_array.flatten()
        p = p / sum(p)
        idx = np.random.choice(range(len(p)), size=n, p=p)

        # build mask of effect centers
        eff_mask = np.zeros(shape)
        ijk_list = [np.unravel_index(_idx, shape) for _idx in idx]
        for ijk in ijk_list:
            eff_mask[ijk] = 1

        # build mask
        mask = prior_array
        if seg_array is not None:
            # build generator of mask per effect center
            mask_gen = (seg_array == seg_array[ijk] for ijk in ijk_list)

            # take union of regions which are valid
            reg_idx_list = reduce(np.logical_or, mask_gen)

            # take intersection with prior_array
            mask = np.logical_and(mask, reg_idx_list)

        # perform dilation
        eff_mask = binary_dilation(eff_mask,
                                   iterations=radius,
                                   mask=mask)
        return Mask(eff_mask.astype(int))


class EffectDm:
    """ dummy effect, doesn't do anything to img, stands in Effect """

    def __len__(self):
        return 0

    def apply(self, x):
        return x

    def apply_from_to_nii(self, f_nii_dict, f_nii_dict_out=None):
        if f_nii_dict_out is not None:
            for feat, f in f_nii_dict.items():
                os.symlink(f, f_nii_dict_out[feat])

        return f_nii_dict_out

    def apply_to_file_tree(self, file_tree):
        return file_tree
