from copy import copy

import numpy as np
from scipy.spatial.distance import dice
from scipy.stats import mannwhitneyu, multivariate_normal

from arba.region import FeatStat
from arba.space import Mask


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

        t2 (float): t squared distance
        u (np.array): effect direction
    """
    @property
    def d(self):
        return len(self.mean)

    @staticmethod
    def from_fs_t2(fs, t2, mask, u=None):
        """ scales effect with observations

        Args:
            fs (FeatStat): stats of affected area
            t2 (float): ratio of effect to population variance
            mask (Mask): effect location
            u (array): direction of offset
        """

        if t2 < 0:
            raise AttributeError('t2 must be positive')

        # get direction u
        if u is None:
            u = draw_random_u(d=fs.d)
        elif len(u) != fs.d:
            raise AttributeError('direction offset must have same len as fs.d')

        # build effect with proper direction, scale to proper t2
        # (ensure u is copied so we have a spare to validate against)
        eff = Effect(mask=mask, mean=copy(u), fs=fs)
        eff.t2 = t2

        assert np.allclose(eff.u, u), 'direction error'
        assert np.allclose(eff.t2, t2), 't2 scale error'

        return eff

    @property
    def t2(self):
        return self.mean @ self.fs.cov_inv @ self.mean

    @t2.setter
    def t2(self, val):
        self.mean *= np.sqrt(val / self.t2)

    @property
    def u(self):
        return self.mean / np.linalg.norm(self.mean)

    @u.setter
    def u(self, val):
        c = val @ self.fs.cov_inv @ val
        self.mean = np.atleast_1d(val) * self.t2 / c

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

        todo: error if estimate has 1's outside of mask
        """
        mask = mask.astype(bool)
        signal = self.mask[mask].astype(bool)
        estimate = estimate[mask].astype(bool)

        if not estimate.sum():
            return 0, 1

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
        """
        file_tree.add(self.mean, mask=self.mask)
