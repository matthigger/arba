import pathlib
import random
from abc import ABC, abstractmethod
from bisect import bisect_right, bisect_left

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from tqdm import tqdm

from arba.plot import save_fig
from arba.space import Mask
from mh_pytools import file, parallel


class PermuteBase(ABC):
    """ runs permutation testing to get maximum 'stat' per region

    # todo: adaptive mode, run only as many n as needed to ensure sig

    Attributes:
        file_tree (FileTree): file_tree
        split_stat_dict (dict): keys are splits, values are maximum stats under
                                that split
    """
    # toggles whether stat of interest is maximized or minimized
    flag_max = True

    # name of save file for object, must be consistent for pre-load (see init)
    f_save = 'permute.p.gz'

    def __init__(self, file_tree, folder=None):
        """

        Args:
            folder (str or Path): path where previous permutation were run (on
                                  same file_tree).  allows one to pick up where
                                  they left off if they want to extend n
        """
        self.file_tree = file_tree
        self.stat_volume = None
        self.pval = None
        self.split_stat_dict = dict()

        if folder is not None:
            # pre load last session's split_stat_dict
            f = folder / self.f_save
            if not f.exists():
                return
            last_self = file.load(f)

            # check that file trees are identical
            assert self.file_tree == last_self.file_tree, \
                'previous splits invalid'

            # use old split_stat_dict
            self.split_stat_dict = last_self.split_stat_dict

    def run(self, split, n=5000, folder=None, seed=1, **kwargs):
        """ runs permutation testing

        Args:
            split (np.array): split to test significance of
            n (int): number of permutations
            folder (str or Path): saves output
            seed (hashable): seed for random num generator
        """
        # set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # no need to repeat a split which is already done
        n -= len(self.split_stat_dict)

        # stop if number of permutations already complete
        if n <= 0:
            return

        with self.file_tree.loaded():
            # build list of new splits
            split_list = list()
            while len(split_list) < n:
                _split = split.sample()
                if _split not in self.split_stat_dict.keys():
                    split_list.append(_split)

            # run splits (permutation sampling from null hypothesis)
            self.run_split_max_multi(split_list=split_list, **kwargs)

            # determine sig
            self.stat_volume, self.pval = self.determine_sig(split)

            # save if folder passed
            if folder is not None:
                self.save(folder=folder, split=split, **kwargs)

    def save(self, folder, split, print_image=False, save_self=False,
             print_hist=False, print_region=False, alpha=.05, **kwargs):
        """ saves output images in a folder"""
        folder = pathlib.Path(folder)
        folder.mkdir(exist_ok=True, parents=True)

        affine = self.file_tree.ref.affine
        for label, x in (('pval', self.pval),
                         ('stat_volume', self.stat_volume)):
            img = nib.Nifti1Image(x, affine)
            img.to_filename(str(folder / f'{label}.nii.gz'))

        folder = pathlib.Path(folder)
        if save_self:
            file.save(self, folder / self.f_save)

        # compute critical stat
        f_out = folder / 'thresh.txt'
        stat_sorted = sorted(self.split_stat_dict.values())
        num_perm = len(stat_sorted)
        if self.flag_max:
            perc_thresh = 100 - alpha * 100
            thresh = np.percentile(stat_sorted, perc_thresh,
                                   interpolation='lower')
        else:
            perc_thresh = alpha * 100
            thresh = np.percentile(stat_sorted, perc_thresh,
                                   interpolation='higher')
        with open(str(f_out), 'w') as f:
            print(f'critical stat thresh: {thresh:.3e}', file=f)

        # built plot of stats observed
        if print_hist:
            sns.set(font_scale=1.2)
            plt.hist(stat_sorted, bins=20)
            plt.gca().axvline(thresh, color='r', label=f'95%={thresh:.2f}')
            plt.gca().legend()
            plt.xlabel(r'extrema stat in Hierarchy')
            plt.ylabel(f'Count\n({num_perm} Permutations)')
            save_fig(folder / 'hist_extrema_stat.pdf', size_inches=(4, 3))

        if print_image:
            # print mean image per grp per feature
            for feat in self.file_tree.feat_list:
                for grp, sbj_bool in split.bool_iter():
                    f_out = folder / f'{feat}_{grp}.nii.gz'
                    self.file_tree.to_nii(feat=feat, sbj_bool=sbj_bool,
                                          f_out=f_out)

        if print_region:
            pval_list = sorted(np.unique(self.pval[self.file_tree.mask]))

            reg_idx = 0
            for p in pval_list:

                if p > alpha:
                    break

                # label connected compoenents of img
                mask = label(pval_list == p)

                p_idx = 1
                while True:
                    _mask = mask == p_idx
                    if not _mask.any():
                        # no more regions @ this pval
                        break

                    # print contiguous region @ this pval
                    _mask = Mask(_mask, ref=self.file_tree.ref)
                    _mask.to_nii(f_out=folder / f'region_{reg_idx}.nii.gz')
                    reg_idx += 1
                    p_idx += 1

    def run_split_max_multi(self, split_list, par_flag=False, verbose=False,
                            effect_split_dict=None, **kwargs):
        """ runs many splits, potentially in parallel"""

        if par_flag:
            arg_list = list()
            for split in split_list:
                arg_list.append({'split': split,
                                 'verbose': False,
                                 'effect_split_dict': effect_split_dict})
            res = parallel.run_par_fnc(fnc='run_split_xtrm', obj=self,
                                       arg_list=arg_list, verbose=verbose)
            for split, xtrm_stat in res:
                self.split_stat_dict[split] = xtrm_stat
        else:
            tqdm_dict = {'disable': not verbose,
                         'desc': 'compute extrema stat per split'}
            for split in tqdm(split_list, **tqdm_dict):
                _, self.split_stat_dict[split] = \
                    self.run_split_xtrm(split,
                                        effect_split_dict=effect_split_dict)

        return self.split_stat_dict

    def run_split_xtrm(self, split, **kwargs):
        """ runs a single split, returns extrema stat (see max_flag)
        """
        stat_volume = self.run_split(split, **kwargs)

        if self.max_flag:
            xtrm = stat_volume[self.file_tree.mask].max()
        else:
            xtrm = stat_volume[self.file_tree.mask].min()

        return split, xtrm

    @abstractmethod
    def run_split(self, split, **kwargs):
        """ returns a volume of statistics per some method

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2) stats
        """

    def determine_sig(self, split=None, stat_volume=None):
        """ runs on the original case, uses the stats saved to determine sig"""
        assert (split is None) != (stat_volume is None), \
            'split xor stat_volume'

        if stat_volume is None:
            # get stat volume of original
            stat_volume = self.run_split(split)

        if self.flag_max:
            bisect = bisect_right
        else:
            bisect = bisect_left

        # build array of pval
        # https://stats.stackexchange.com/questions/109207/p-values-equal-to-0-in-permutation-test
        # todo: add confidence from CLT approx of binmoial
        stat_sorted = sorted(self.split_stat_dict.values())
        n = len(self.split_stat_dict)
        pval = np.zeros(self.file_tree.ref.shape)
        for i, j, k in self.file_tree.pc:
            p = bisect(stat_sorted, stat_volume[i, j, k]) / n
            if self.flag_max:
                p = 1 - p
            pval[i, j, k] = p

        return stat_volume, pval

    def get_t2(self, split, verbose=False):
        """ computes t2 stat per voxel (not scaled) """

        raise NotImplementedError('scale?')

        # build fs per ijk in mask
        t2 = np.zeros(self.file_tree.ref.shape)
        tqdm_dict = {'disable': not verbose,
                     'desc': 'compute t2 per vox'}
        sbj_bool = split.sbj_bool
        for ijk in tqdm(self.file_tree.pc, **tqdm_dict):
            fs0 = self.file_tree.get_fs(ijk=ijk,
                                        sbj_bool=np.logical_not(sbj_bool))
            fs1 = self.file_tree.get_fs(ijk=ijk,
                                        sbj_bool=sbj_bool)

            # compute t2
            delta = fs0.mu - fs1.mu
            pool_cov = fs0.get_pool_cov((fs0, fs1))
            i, j, k = ijk
            t2[i, j, k] = delta @ np.linalg.inv(pool_cov) @ delta

        return t2
