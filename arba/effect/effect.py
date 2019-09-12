import numpy as np
from scipy.spatial.distance import dice
from scipy.stats import mannwhitneyu

from arba.region import FeatStat
from arba.space import Mask
from .get_sens_spec import get_sens_spec


class Effect:
    """ adds an effect to an image

    an effect is a constant offset to a set of voxels, scale may vary

    Attributes:
        mask (Mask): effect location
        fs (FeatStat): FeatStat of unaffected area (used to compute severity
                       of an effect ... eg t2)
        scale (np.array): scale of effect, defaults to mask, otherwise values
                  between 0 and 1. allows `soft' boundary to effect

    todo: call of get_auc(), get_dice() and get_sens_spec() should be uniform
    """

    def __init__(self, mask, scale=None, fs=None):
        self.mask = mask
        self.fs = fs
        self.scale = scale
        if self.scale is None:
            self.scale = self.mask

    def __len__(self):
        return len(self.mask)

    def get_auc(self, x, mask):
        """ computes auc of statistic given by array feat

        Args:
            x (np.array): scores (per voxel)
            mask (mask): values in feat which are to be counted towards auc

        Returns:
            auc (float): value in [0, 1]
        """
        # mask the statistic
        stat_vec = np.array(x[mask.astype(bool)])

        # mask the ground truth to relevant area
        truth_vec = np.array(self.mask[mask.astype(bool)])

        # compute feat, y
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

    def get_sens_spec(self, estimate, mask=None):
        get_sens_spec(target=self.mask, estimate=estimate, mask=mask)
