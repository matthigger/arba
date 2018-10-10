from functools import reduce

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

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
            f_img_tree (dict of dict):
            mask (Mask): effect location
            effect_snr (float): ratio of effect to population variance
            cov_ratio (float): ratio of effect cov to population cov
        """

        sbj = next(iter(f_img_tree.keys()))
        feat_label = sorted(f_img_tree[sbj].keys())

        # get mean and cov of all img corresponding to mask
        fs_list = list()
        for sbj, f_img_dict in f_img_tree.items():
            x = np.hstack(mask.apply_from_nii(f_img_dict[label])
                          for label in feat_label)
            fs = FeatStat.from_array(x)
            fs_list.append(fs)
        fs = sum(fs_list)

        #
        mean = fs.mu * effect_snr
        cov = fs.cov * cov_ratio

        return Effect(mask, mean, cov, feat_label=feat_label)

    def __init__(self, mask, feat_label, mean, cov=None):
        if not isinstance(mask, Mask):
            raise TypeError(f'mask: {mask} must be of type Mask')
        self.mask = mask
        self.mean = np.atleast_1d(mean)
        self.cov = np.atleast_2d(cov)
        self.feat_label = feat_label

    def __len__(self):
        return len(self.mask)

    def apply_to_nii(self, f_nii_dict=None):
        def load(f_nii):
            """ loads f_nii, ensures proper space
            """
            img = nib.load(str(f_nii))
            if self.mask.affine is not None and \
                    not np.array_equal(img.affine, self.mask.affine):
                raise AttributeError('space mismatch')
            return img.get_data()

        # load and stack data
        x_list = [load(f_nii_dict[label]) for label in self.feat_label]
        x = np.stack(x_list, axis=len(x_list[0].shape))

        return self.apply(x)

    def apply(self, x):
        """ applies effect to array x

        Args:
            x (np.array): img to apply effect to
        """

        # sample effect
        effect = np.vstack(np.random.multivariate_normal(mean=self.mean,
                                                         cov=self.cov,
                                                         size=len(self))).T

        raise NotImplementedError('check that unraveling in proper order')
        return self.mask.insert(x, effect, add=True)

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
