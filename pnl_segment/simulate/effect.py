from copy import copy
from functools import reduce

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import dice
from scipy.stats import mannwhitneyu, multivariate_normal

from ..region import FeatStat
from ..space import Mask, PointCloud


def draw_random_u(d):
    """ Draws random vector in d dimensional unit sphere

    Args:
        d (int): dimensionality of vector
    Returns:
        u (np.array): random vector in d dim unit sphere
    """
    mu = np.zeros(d)
    cov = np.eye(d)
    u = multivariate_normal.rvs(mean=mu, cov=cov)
    return u / np.linalg.norm(u)


class Effect:
    """ adds an effect to an image

    note: only iid normal effects are supported

    Attributes:
        mean (np.array): average offset of effect on a voxel
        mask (Mask): effect location
        fs (FeatStat): FeatStat of unaffected area

        maha (float): mahalanobis
        u (np.array): effect direction
    """

    @staticmethod
    def from_fs_maha(fs, maha, mask, u=None):
        """ scales effect with observations

        Args:
            fs (FeatStat): stats of affected area
            maha (float): ratio of effect to population variance
            mask (Mask): effect location
            u (array): direction of offset
        """

        if maha < 0:
            raise AttributeError('maha must be positive')

        # get direction u
        if u is None:
            u = draw_random_u(d=fs.d)
        elif len(u) != fs.d:
            raise AttributeError('direction offset must have same len as fs.d')

        # build effect with proper direction, scale to proper maha
        # (ensure u is copied so we have a spare to validate against)
        eff = Effect(mask=mask, mean=copy(u), fs=fs)
        eff.maha = maha

        assert np.allclose(eff.u, u), 'direction error'
        assert np.allclose(eff.maha, maha), 'maha scale error'

        return eff

    @property
    def maha(self):
        return np.sqrt(self.mean @ self.fs.cov_inv @ self.mean)

    @maha.setter
    def maha(self, val):
        self.mean *= val / self.maha

    @property
    def u(self):
        return self.mean / np.linalg.norm(self.mean)

    @u.setter
    def u(self, val):
        c = val @ self.fs.cov_inv @ val
        self.mean = np.array(val) * self.maha / c

    def __init__(self, mask, mean, fs):
        self.mask = mask
        self.mean = np.reshape(mean, len(mean))
        self.fs = fs

    def __len__(self):
        return len(self.mask)

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

    def get_sens_spec(self, estimate, mask):
        """ returns sens + spec

        Args:
            estimate (np.array): boolean array, estimate of effect location
            mask (np.array): boolean array, voxels outside of mask are not
                             counted for or against accuracy

        Returns:
            sens (float): percentage of affected voxels detected
            spec (float): percentage of unaffected voxels undetected
        """
        mask = mask.astype(bool)
        signal = self.mask[mask].astype(bool)
        estimate = estimate[mask].astype(bool)

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

    def apply_to_file_tree(self, file_tree):
        ijk_set = PointCloud.from_mask(self.mask)
        for ijk in ijk_set:
            file_tree.ijk_fs_dict[ijk].mu += self.mean

    @staticmethod
    def sample_mask(prior_array, radius, n=1, seg_array=None, ref=None):
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
        p = np.array(prior_array / prior_array.sum())
        idx = np.random.choice(range(p.size), size=n, p=p.flatten())

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
        return Mask(eff_mask, ref=ref)
