import numpy as np
from tqdm import tqdm

from .compute import apply_tfce
from ...permute_base import PermuteBase


class PermuteTFCE(PermuteBase):
    def __init__(self, *args, h=2, e=.5, c=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.e = e
        self.c = c

    def run_split(self, split, **kwargs):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        t2 = self.get_t2(split)
        t2_tfce = apply_tfce(t2, h=self.h, e=self.e, c=self.c)
        return t2_tfce

    def get_t2(self, split, verbose=False):
        """ computes t2 stat per voxel (not scaled) """

        split = np.array(split)

        # build fs per ijk in mask
        t2 = np.zeros(self.file_tree.ref.shape)
        tqdm_dict = {'disable': not verbose,
                     'desc': 'compute t2 per vox'}
        split_not = tuple([not x for x in split])
        for ijk in tqdm(self.file_tree.pc, **tqdm_dict):
            fs0 = self.file_tree.get_fs(ijk=ijk, sbj_bool=split_not)
            fs1 = self.file_tree.get_fs(ijk=ijk, sbj_bool=split)

            # compute t2
            delta = fs0.mu - fs1.mu
            cov_pooled = (fs0.cov * fs0.n +
                          fs1.cov * fs1.cov) / (fs0.n + fs1.n)
            i, j, k = ijk
            t2[i, j, k] = delta @ np.linalg.inv(cov_pooled) @ delta

        return t2
