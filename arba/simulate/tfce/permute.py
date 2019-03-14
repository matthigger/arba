import numpy as np
from tqdm import tqdm

from .compute import apply_tfce
from ...permute import Permute
from ...region import FeatStat


class PermuteTFCE(Permute):
    def __init__(self, *args, h=2, e=.5, c=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.e = e
        self.c = c

    def run_split(self, x, split):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            x (np.array): (space0, space1, space2, num_sbj, num_feat)
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        t2 = self.get_t2(x, split)
        t2_tfce = apply_tfce(t2, h=self.h, e=self.e, c=self.c)
        return t2_tfce

    def get_t2(self, x, split, verbose=False):
        """ computes t2 stat per voxel (not scaled) """

        split = np.array(split)

        # build fs per ijk in mask
        t2 = np.zeros(self.pc.ref.shape)
        tqdm_dict = {'disable': not verbose,
                     'desc': 'compute t2 per vox'}
        for i, j, k in tqdm(self.pc, **tqdm_dict):
            # get feat stat of each grp
            fs0 = FeatStat.from_array(x[i, j, k, split == 0, :].T, _fast=True)
            fs1 = FeatStat.from_array(x[i, j, k, split == 1, :].T, _fast=True)

            # compute t2
            delta = fs0.mu - fs1.mu
            cov_pooled = (fs0.cov * fs0.n +
                          fs1.cov * fs1.cov) / (fs0.n + fs1.n)
            t2[i, j, k] = delta @ np.linalg.inv(cov_pooled) @ delta

        return t2
