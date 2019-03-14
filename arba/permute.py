from abc import ABC, abstractmethod
from bisect import bisect_right

import numpy as np

from arba.space import PointCloud


class Permute(ABC):
    """ runs permutation testing to get maximum 'stat' per region

    # todo: adaptive mode, run only as many n as needed to ensure sig

    a permutation is characterized by a split, a tuple of booleans.  split[i]
    describes if the i-th subject is in grp 1 (under that split)


    note: this implementation is heavy on indexing. given the importance, we
    store FileTrees and their labels as tuples (rather than a dict)

    Attributes:
        grp_tuple (tuple): labels of each file tree
        ft_tuple (tuple): FileTrees of data
        folder (Path): path to output
        split_stat_dict (dict): keys are splits, values are maximum stats under
                                that split
    """

    @property
    def split(self):
        ft0, ft1 = self.ft_tuple
        return np.hstack((np.zeros(len(ft0)),
                          np.ones(len(ft1))))

    def __init__(self, ft_dict, folder=None):
        (grp0, ft0), (grp1, ft1) = sorted(ft_dict.items())

        assert ft0.mask == ft1.mask, 'space mismatch'
        assert not (set(ft0.sbj_list) & set(ft1.sbj_list)), 'sbj intersection'

        self.ft_tuple = (ft0, ft1)
        self.grp_tuple = (grp0, grp1)
        self.pc = PointCloud.from_mask(ft0.mask)

        if folder is not None:
            raise NotImplementedError('not sure how to do this just yet')
        else:
            self.split_stat_dict = dict()

    def run(self, n=5000, folder=None, **kwargs):
        """ runs permutation testing

        Args:
            n (int): number of permutations
        """
        # no need to repeat a split which is already done
        n = max(n - len(self.split_stat_dict), 0)

        if n > 0:
            # build list of splits
            split_list = self.sample_splits(n, **kwargs)

            # load data
            x = self.load_data(**kwargs)

            # run splits (permutation sampling from null hypothesis)
            self.run_split_max_multi(x, split_list, **kwargs)

        # determine sig
        stat_volume, pval = self.determine_sig(x)

        # save if folder passed
        if folder is not None:
            self.save(stat_volume, pval)

        return stat_volume, pval

    def load_data(self, **kwargs):
        """ get data matrix """
        ft0, ft1 = self.ft_tuple

        for ft in self.ft_tuple:
            ft.load(**kwargs, load_ijk_fs=False)

        x = np.concatenate((ft0.data, ft1.data), axis=3)

        for ft in self.ft_tuple:
            ft.unload()

        return x

    def sample_splits(self, n, seed=1, **kwargs):
        """ sample splits, ensure none are repeated

        note: each split has same number of ones as self.split

        Args:
            n (int): number of permutations
            seed : initialize random number generator (helpful for debug)

        Returns:
            split_list (list): each element is a split
        """
        ft0, ft1 = self.ft_tuple
        num_ones = len(ft1)
        num_sbj = len(ft0) + len(ft1)

        np.random.seed(seed)

        split_list = list()
        while len(split_list) < n:
            # build a split
            ones_idx = np.random.choice(range(num_sbj), size=num_ones,
                                        replace=False)
            split = tuple(idx in ones_idx for idx in range(num_sbj))

            # add this split so long as its not already in the split_stat_dict
            if split not in self.split_stat_dict.keys():
                split_list.append(split)

        return split_list

    def run_split_max_multi(self, x, splits, par_flag=False, **kwargs):
        """ runs many splits, potentially in parallel"""

        if par_flag:
            raise NotImplementedError
        else:
            for split in splits:
                self.split_stat_dict[split] = self.run_split_max(x, split)

    def run_split_max(self, x, split):
        """ runs a single split, returns max stat """
        stat_volume = self.run_split(x, split)

        # return max in mask
        mask = self.ft_tuple[0].mask
        return stat_volume[mask].max()

    @abstractmethod
    def run_split(self, x, split):
        """ returns a volume of statistics per some method

        Args:
            x (np.array): (space0, space1, space2, num_sbj, num_feat)
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2) stats
            pval (np.array): (space0, space1, space2) pval (corrected)
        """

    def determine_sig(self, x):
        """ runs on the original case, uses the stats saved to determine sig"""

        # get stat volume of original
        stat_volume = self.run_split(x, self.split)

        # build array of pval
        # https://stats.stackexchange.com/questions/109207/p-values-equal-to-0-in-permutation-test
        # todo: add confidence from CLT approx of binmoial
        stat_sorted = sorted(self.split_stat_dict.values())
        n = len(self.split_stat_dict)
        pval = np.zeros(x.shape[:3])
        for i, j, k in self.pc:
            p = bisect_right(stat_sorted, stat_volume[i, j, k]) / n
            pval[i, j, k] = 1 - p

        return stat_volume, pval
