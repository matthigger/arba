import random
from collections import Counter
from copy import copy

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import dice
from scipy.stats import mannwhitneyu, multivariate_normal
from skimage import measure

from ..region import FeatStat
from ..space import Mask, PointCloud, get_ref


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
        self.mean = np.atleast_1d(val) * self.maha / c

    def __init__(self, mask, mean, fs):
        self.mask = mask
        self.mean = np.atleast_1d(mean)
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
        """ applies effect to a file tree

        NOTE: we avoid building apply_to_array() and apply_to_file() methods
        directly as effect may be multivariate; file_tree encapsulates the
        indexing of effects.
        """
        ijk_set = PointCloud.from_mask(self.mask)
        for ijk in ijk_set:
            file_tree.ijk_fs_dict[ijk].mu += self.mean

        # apply to raw data
        _mask = np.broadcast_to(self.mask.T, file_tree.data.T.shape).T
        file_tree.data = np.add(file_tree.data, self.mean, where=_mask)

    @staticmethod
    def sample_mask(prior_array, radius=None, seg_array=None, ref=None,
                    num_vox=None):
        """n effect centers chosen and dilated to build mask of affected volume

        Args:
            prior_array (np.array): has same shape as image.  values are
                                    unweighted prob of effect center.
                                    note that prior array also masks effect (no
                                    effect exists where prior_array <= 0)
            radius (int, tuple): if int, acts as effect radius: center is
                                 dilated this many times.  if tuple, expected
                                 as shape of effect
            seg_array (np.array): segmentation array, if passed constrains
                                  effect to the region of its center. has
                                  same shape as prior_array
            ref : reference space of output mask
            num_vox (int): if passed, will ensure mask has exactly total_vox
                             points

        Returns:
            eff_mask (np.array): array of 1s where effect is present
        """
        # check that inputs are consistent
        assert (radius is None) != (num_vox is None), \
            'radius xor num_vox required'

        # if num_vox passed, there are constraints on prior_array and seg_array
        if num_vox is not None:
            if seg_array is not None:
                # avoid memory intersection with prior_array passed
                prior_array = copy(prior_array)

                # selection must be in intersection of seg_array + prior_array
                prior_array[np.logical_not(seg_array)] = 0
                seg_array[np.logical_not(prior_array)] = 0

                # ensure each idx of seg_array is connected
                seg_array = measure.label(seg_array)

                # rm idx in seg_array which don't have at least num_vox voxels
                c = Counter(seg_array.flatten())
                idx_invalid = [idx for idx, count in c.items()
                               if count < num_vox]
                assert len(idx_invalid) < len(c), \
                    f'seg_array doesnt have regions with {num_vox} voxels'
                for idx in idx_invalid:
                    prior_array[seg_array == idx] = 0

            assert prior_array.astype(bool).sum() >= num_vox, \
                f'prior_array doesnt have {num_vox} valid voxels'

        # choose effect center via prior array
        p = np.array(prior_array / prior_array.sum())
        idx = np.random.choice(range(p.size), size=1, p=p.flatten())
        center_ijk = np.unravel_index(idx[0], prior_array.shape)

        # init mask
        mask = np.zeros(prior_array.shape)
        mask[center_ijk] = 1

        # meta_mask is a mask of the mask (contains all the possible voxels
        # that an effect mask could contain)
        meta_mask = prior_array.astype(bool)
        if seg_array is not None:
            meta_mask = np.logical_and(meta_mask,
                                       seg_array == seg_array[center_ijk])

        # perform dilation
        if radius is not None:
            if isinstance(radius, int):
                # dilate radius number of times
                mask = binary_dilation(mask, iterations=radius, mask=meta_mask)
            elif len(radius) == 3:
                # shape specified
                for delta in np.ndindex(radius):
                    mask[tuple(center_ijk + delta)] = 1
                mask = np.logical_and(mask, meta_mask)
            else:
                raise AttributeError('invalid radius, must be int or tuple')
            return Mask(mask, ref=get_ref(ref))

        # dilate mask until it has > num_vox voxels
        assert num_vox > 1, 'num_vox must be > 1'
        while mask.sum() < num_vox:
            mask_last = mask
            mask = binary_dilation(mask, iterations=1, mask=meta_mask,
                                   structure=np.ones((3, 3, 3)))

        # rm some of the voxels added in last dilation until mask has num_vox
        shell_mask = Mask(np.logical_and(mask, np.logical_not(mask_last)))
        shell_pc = PointCloud.from_mask(shell_mask)
        ijk_to_rm = random.sample(shell_pc, k=mask.sum() - num_vox)
        for ijk in ijk_to_rm:
            mask[ijk] = False

        assert mask.sum() == num_vox, 'mask doesnt have num_vox voxels'

        return Mask(mask, ref=get_ref(ref))
