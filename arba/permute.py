import os
import pathlib
from abc import ABC, abstractmethod
from bisect import bisect_right

import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba.space import PointCloud
from mh_pytools import file, parallel


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
        split_stat_dict (dict): keys are splits, values are maximum stats under
                                that split
        pc (PointCloud): all active voxels
    """
    f_save = 'permute.p.gz'

    @property
    def split(self):
        ft0, ft1 = self.ft_tuple
        return np.hstack((np.zeros(len(ft0)),
                          np.ones(len(ft1))))

    def __init__(self, ft_dict, folder=None):
        """

        Args:
            ft_dict (dict): keys are grp labels, values are FileTree
            folder (str or Path): path where previous permutation were run (on
                                  same ft_dict).  allows one to pick up where
                                  they left off if they want to extend n
        """
        (grp0, ft0), (grp1, ft1) = sorted(ft_dict.items())

        assert ft0.mask == ft1.mask, 'space mismatch'
        assert not (set(ft0.sbj_list) & set(ft1.sbj_list)), 'sbj intersection'

        self.ft_tuple = (ft0, ft1)
        self.grp_tuple = (grp0, grp1)
        self.pc = PointCloud.from_mask(ft0.mask)

        self.split_stat_dict = dict()

        if folder is not None:
            f = folder / self.f_save
            if not f.exists():
                return
            last_self = file.load(f)

            # check that file trees are identical
            assert self.ft_tuple == last_self.ft_tuple, 'previous splits invalid'

            # use old split_stat_dict
            self.split_stat_dict = last_self.split_stat_dict

    def run(self, n=5000, folder=None, **kwargs):
        """ runs permutation testing

        Args:
            n (int): number of permutations
            folder (str or Path): saves output
        """
        # no need to repeat a split which is already done
        n = max(n - len(self.split_stat_dict), 0)

        # load data
        x = self.load_data(**kwargs)

        if n > 0:
            # build list of splits
            split_list = self.sample_splits(n, **kwargs)

            # run splits (permutation sampling from null hypothesis)
            self.run_split_max_multi(x=x, split_list=split_list, **kwargs)

        # determine sig
        stat_volume, pval = self.determine_sig(x)

        # save if folder passed
        if folder is not None:
            label_x_dict = {'stat_volume': stat_volume,
                            'pval': pval}
            self.save(folder, label_x_dict)

        return stat_volume, pval

    def save(self, folder, label_x_dict=None):
        """ saves output images in a folder"""
        file.save(self, pathlib.Path(folder) / self.f_save)

        if label_x_dict is None:
            return

        affine = self.ft_tuple[0].ref.affine
        for label, x in label_x_dict.items():
            img = nib.Nifti1Image(x, affine)
            img.to_filename(str(folder / f'{label}.nii.gz'))

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

    def run_split_max_multi(self, split_list, x=None, f_x=None, par_flag=False,
                            verbose=False,
                            **kwargs):
        """ runs many splits, potentially in parallel"""

        if not par_flag:
            assert (x is None) != (f_x is None), 'either x xor f_x required'

            if f_x is not None:
                x = file.load(f_x)

            tqdm_dict = {'disable': not verbose,
                         'desc': 'compute max stat per split'}
            for split in tqdm(split_list, **tqdm_dict):
                self.split_stat_dict[split] = self.run_split_max(x, split)

            return self.split_stat_dict

        arg_list = list()

        assert x is not None, 'x required in parallel mode'
        f_x = file.save(x)

        num_cpu = min(parallel.num_cpu, len(split_list))
        idx = np.linspace(0, len(split_list), num=num_cpu + 1).astype(int)
        for cpu_idx in range(num_cpu):
            # only verbose on last cpu
            _verbose = verbose and (cpu_idx == num_cpu - 1)

            arg_list.append({'f_x': f_x,
                             'split_list': split_list[idx[cpu_idx]:
                                                      idx[cpu_idx + 1]],
                             'par_flag': False,
                             'verbose': _verbose})
        res = parallel.run_par_fnc(fnc='run_split_max_multi', obj=self,
                                   arg_list=arg_list, verbose=verbose)
        for _split_stat_dict in res:
            self.split_stat_dict.update(_split_stat_dict)

        os.remove(str(f_x))
        return self.split_stat_dict

    def run_split_max(self, x, split):
        """ runs a single split, returns max stat """
        stat_volume = self.run_split(x, split)

        # return max in mask
        mask = self.ft_tuple[0].mask
        return stat_volume[mask].max()

    @abstractmethod
    def run_split(self, x, split):
        """ returns a volume of statistics per some method

        NOTE: in multiprocessing x is a memmap which can only be read from

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
