import pathlib
from abc import ABC, abstractmethod
from bisect import bisect_right

import nibabel as nib
import numpy as np
from scipy.special import comb
from tqdm import tqdm

from mh_pytools import file, parallel
import matplotlib.pyplot as plt
import seaborn as sns
from arba.plot import save_fig


class Permute(ABC):
    """ runs permutation testing to get maximum 'stat' per region

    # todo: adaptive mode, run only as many n as needed to ensure sig

    a permutation is characterized by a split, a tuple of booleans.  split[i]
    describes if the i-th subject is in grp 1 (under that split)


    Attributes:
        file_tree (FileTree): file_tree
        split_stat_dict (dict): keys are splits, values are maximum stats under
                                that split
    """
    f_save = 'permute.p.gz'

    def __init__(self, file_tree, folder=None):
        """

        Args:
            folder (str or Path): path where previous permutation were run (on
                                  same file_tree).  allows one to pick up where
                                  they left off if they want to extend n
        """
        self.file_tree = file_tree

        self.split_stat_dict = dict()

        if folder is not None:
            f = folder / self.f_save
            if not f.exists():
                return
            last_self = file.load(f)

            # check that file trees are identical
            assert self.file_tree == last_self.file_tree, \
                'previous splits invalid'

            # use old split_stat_dict
            self.split_stat_dict = last_self.split_stat_dict

    def run(self, split, n=5000, folder=None, **kwargs):
        """ runs permutation testing

        Args:
            split (np.array): split to test significance of
            n (int): number of permutations
            folder (str or Path): saves output
        """
        # no need to repeat a split which is already done
        n = max(n - len(self.split_stat_dict), 0)

        with self.file_tree.loaded():
            if n > 0:
                # build list of splits
                split_list = self.sample_splits(n, split, **kwargs)

                # run splits (permutation sampling from null hypothesis)
                self.run_split_max_multi(split_list=split_list, **kwargs)

            # determine sig
            stat_volume, pval = self.determine_sig(split)

        # save if folder passed
        if folder is not None:
            label_x_dict = {'stat_volume': stat_volume,
                            'pval': pval}
            self.save(folder=folder, label_x_dict=label_x_dict, split=split,
                      **kwargs)

        return stat_volume, pval

    def save(self, folder, split=None, label_x_dict=None, print_image=True,
             **kwargs):
        """ saves output images in a folder"""
        folder = pathlib.Path(folder)
        file.save(self, folder / self.f_save)

        # built plot of stats observed
        sns.set(font_scale=1.2)
        stats = np.array(list(self.split_stat_dict.values()))
        bins = np.geomspace(stats.min(), stats.max(), 100)
        thresh = np.percentile(stats, 95)
        plt.hist(stats, bins=bins)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.gca().axvline(thresh, color='r', label=f'95%={thresh}')
        plt.legend()
        plt.xlabel('max stat across image')
        plt.ylabel(f'count\n({len(stats)} splits total)')
        save_fig(folder / 'hist_max_stat.pdf')

        if print_image:
            assert split is not None, 'split required if print_image'

            # print mean image per grp per feature
            not_split = tuple(not x for x in split)
            for feat in self.file_tree.feat_list:
                for lbl, split in (('1', split),
                                   ('0', not_split)):
                    f_out = folder / f'{feat}_{lbl}.nii.gz'
                    self.file_tree.to_nii(feat=feat, sbj_bool=split,
                                          f_out=f_out)

        if label_x_dict is None:
            return

        affine = self.file_tree.ref.affine
        for label, x in label_x_dict.items():
            img = nib.Nifti1Image(x, affine)
            img.to_filename(str(folder / f'{label}.nii.gz'))

    def sample_splits(self, n, split, seed=1, **kwargs):
        """ sample splits, ensure none are repeated

        note: each split has same number of ones as self.split

        Args:
            n (int): number of permutations
            split (np.array): the target split (used to count 0s and 1s)
            seed : initialize random number generator (helpful for debug)

        Returns:
            split_list (list): each element is a split
        """
        num_ones = sum(split)
        num_sbj = len(split)

        assert comb(num_sbj, num_ones) >= (n + len(self.split_stat_dict)), \
            'not enough unique splits'

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

    def run_split_max_multi(self, split_list, par_flag=False, verbose=False,
                            effect_split_dict=None):
        """ runs many splits, potentially in parallel"""

        if par_flag:
            arg_list = list()
            for split in split_list:
                arg_list.append({'split': split,
                                 'verbose': False,
                                 'effect_split_dict': effect_split_dict})
            res = parallel.run_par_fnc(fnc='run_split_max', obj=self,
                                       arg_list=arg_list, verbose=verbose)
            for split, max_stat in res:
                self.split_stat_dict[split] = max_stat
        else:
            tqdm_dict = {'disable': not verbose,
                         'desc': 'compute max stat per split'}
            for split in tqdm(split_list, **tqdm_dict):
                _, self.split_stat_dict[split] = self.run_split_max(split)

        return self.split_stat_dict

    def run_split_max(self, split, **kwargs):
        """ runs a single split, returns max stat """
        stat_volume = self.run_split(split, **kwargs)

        return split, stat_volume[self.file_tree.mask].max()

    @abstractmethod
    def run_split(self, split, effect_split_dict):
        """ returns a volume of statistics per some method

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split
            effect_split_dict (dict): keys are effects, values are splits

        Returns:
            stat_volume (np.array): (space0, space1, space2) stats
            pval (np.array): (space0, space1, space2) pval (corrected)
        """

    def determine_sig(self, split=None, stat_volume=None):
        """ runs on the original case, uses the stats saved to determine sig"""
        assert (split is None) != (stat_volume is None), \
            'split xor stat_volume'

        if stat_volume is None:
            # get stat volume of original
            stat_volume = self.run_split(split)

        # build array of pval
        # https://stats.stackexchange.com/questions/109207/p-values-equal-to-0-in-permutation-test
        # todo: add confidence from CLT approx of binmoial
        stat_sorted = sorted(self.split_stat_dict.values())
        n = len(self.split_stat_dict)
        pval = np.zeros(self.file_tree.ref.shape)
        for i, j, k in self.file_tree.pc:
            p = bisect_right(stat_sorted, stat_volume[i, j, k]) / n
            pval[i, j, k] = 1 - p

        return stat_volume, pval
