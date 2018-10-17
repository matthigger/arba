import shutil
import tempfile
from functools import reduce

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import mannwhitneyu

from pnl_segment.adaptive.feat_stat import FeatStat
from pnl_segment.simulate.mask import Mask


class Effect:
    """ adds an effect to an image

    note: only iid normal effects are supported

    Attributes:
        mean (np.array): average offset of effect on a voxel
        cov (np.array): square matrix, noise power
        mask (Mask): effect location
    """

    @staticmethod
    def from_data(f_img_tree, mask, effect_snr, cov_ratio=0):
        """ scales effect with observations

        Args:
            f_img_tree (dict of dict): leafs can be files or arrays
            mask (Mask): effect location
            effect_snr (float): ratio of effect to population variance
            cov_ratio (float): ratio of effect cov to population cov
        """

        sbj = next(iter(f_img_tree.keys()))
        feat_label = sorted(f_img_tree[sbj].keys())

        if isinstance(next(iter(f_img_tree[sbj].values())), np.ndarray):
            get_data = mask.apply
        else:
            get_data = mask.apply_from_nii

        # get mean and cov of all img corresponding to mask
        fs_list = list()
        for sbj, f_img_dict in f_img_tree.items():
            x = np.hstack(get_data(f_img_dict[label]) for label in feat_label)
            fs = FeatStat.from_array(x)
            fs_list.append(fs)
        fs = sum(fs_list)

        #
        mean = np.diag(fs.cov) * effect_snr
        cov = fs.cov * cov_ratio

        return Effect(mask, mean=mean, cov=cov, feat_label=feat_label)

    def __init__(self, mask, feat_label, mean, cov=None):
        if not isinstance(mask, Mask):
            raise TypeError(f'mask: {mask} must be of type Mask')
        self.mask = mask
        self.mean = np.squeeze(mean)
        if len(self.mean.shape) > 1:
            raise AttributeError('1d mean required')
        self.cov = np.atleast_2d(cov)
        self.feat_label = feat_label

    def __len__(self):
        return len(self.mask)

    def apply_from_to_nii(self, f_nii_dict, f_nii_dict_out=None):
        def load(f_nii):
            """ loads f_nii, ensures proper space
            """
            img = nib.load(str(f_nii))
            if self.mask.ref_space is not None and \
                    not np.array_equal(img.affine, self.mask.ref_space.affine):
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

    def get_auc_from_nii(self, f_nii):
        """ computes auc of statistic given in f_nii

        a strong auc value requires that there exists some threshold which
        seperates affected voxels from unaffected voxels.  here, self serves
        as 'ground truth' of which voxels are, or are not, affected

        Args:
            f_nii (str or Path): path to statistic image

        Returns:
            auc (float): value in [0, 1]
        """
        stat_mask = Mask.from_nii(f_nii)
        stat_vec = stat_mask.apply_from_nii(f_nii)
        truth_vec = stat_mask.apply(self.mask.x)

        x = stat_vec[truth_vec == 0]
        y = stat_vec[truth_vec == 1]
        u = mannwhitneyu(x, y, alternative='greater')
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
        x = self.mask.insert(x, effect, add=True)

        return x

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

    def __init__(self, feat_label, *args, **kwargs):
        self.feat_label = feat_label

    def __len__(self):
        return 0

    def apply(self, x):
        return x

    def apply_from_to_nii(self, f_nii_dict, f_nii_dict_out=None):
        if f_nii_dict_out is not None:
            for feat, f in f_nii_dict.items():
                shutil.copy(f, f_nii_dict_out[feat])

        return f_nii_dict_out
