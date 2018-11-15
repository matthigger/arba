import os
import tempfile
from copy import deepcopy
from functools import reduce

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import dice
from scipy.stats import mannwhitneyu

from ..region import FeatStat
from ..space import Mask, PointCloud


class Effect:
    """ adds an effect to an image

    note: only iid normal effects are supported

    Attributes:
        mean (np.array): average offset of effect on a voxel
        cov (np.array): square matrix, noise power
        mask (Mask): effect location
    """

    @staticmethod
    def from_data(fs, mask, snr, cov_ratio=0, u=None, **kwargs):
        """ scales effect with observations

        Args:
            fs (FeatStat): stats of affected area
            mask (Mask): effect location
            snr (float): ratio of effect to population variance
            cov_ratio (float): ratio of effect cov to population cov
            u (array): direction of offset
        """

        if snr < 0:
            raise AttributeError('snr must be positive')

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

        # compute x, y
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

    def get_dice(self, mask):
        """ computes dice score
        """

        if sum(mask.flatten()):
            return 1 - dice(mask.flatten(), self.mask.flatten())
        else:
            # no area detected
            return 0

    def get_sens_spec(self, mask, mask_active):
        """ returns sens + spec

        Returns:
            sens (float): percentage of affected voxels detected
            spec (float): percentage of unaffected voxels undetected
        """
        signal = self.mask[mask_active].astype(bool)
        estimate = mask[mask_active].astype(bool)

        true_pos = np.count_nonzero(signal & estimate)
        true = np.count_nonzero(signal)

        if true:
            sens = true_pos / true
        else:
            # if nothing to detect, we default to sensitivity 0 (graphing)
            sens = np.nan

        neg = np.count_nonzero(~signal)
        true_neg = np.count_nonzero((~signal) & (~estimate))

        if neg:
            spec = true_neg / neg
        else:
            # if no negative estimates, we default to specificity 0 (graphing)
            spec = np.nan

        return sens, spec

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

        ijk_set = PointCloud.from_mask(self.mask)
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
        if isinstance(radius, int):
            eff_mask = binary_dilation(eff_mask,
                                       iterations=radius,
                                       mask=mask)
        elif len(radius) == 3:
            pc = PointCloud.from_mask(eff_mask)
            ijk = np.array(next(iter(pc)))
            for delta in np.ndindex(radius):
                pc.add(tuple(ijk + delta))
            eff_mask = pc.to_mask(shape=prior_array.shape)
            eff_mask = np.logical_and(eff_mask, prior_array)
        else:
            raise AttributeError(r'invalid radius given: {radius}')
        return Mask(eff_mask)


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
