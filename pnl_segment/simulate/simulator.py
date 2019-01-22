import pathlib
import random

import numpy as np
from tqdm import tqdm

from mh_pytools import parallel
from . import Effect
from ..seg_graph import run_arba
from ..space import PointCloud


def increment_to_unique(folder, num_width=3):
    idx = 0
    while True:
        _folder = pathlib.Path(
            str(folder) + '_run' + str(idx).zfill(num_width))
        if not _folder.exists():
            _folder.mkdir(exist_ok=True, parents=True)
            return _folder
        idx += 1


class Simulator:
    grp_effect = 'grp_effect'
    grp_null = 'grp_null'

    def __init__(self, folder, file_tree, p_effect=.5):
        self.folder = pathlib.Path(folder)
        folder.mkdir(parents=True)

        # split into two file_trees
        self.file_tree = file_tree
        ft_eff, ft_null = file_tree.split(p=p_effect)
        self.ft_dict = {self.grp_effect: ft_eff,
                        self.grp_null: ft_null}

        self.effect_list = list()

    def build_effect_list(self, n_effect, verbose=False, seed=1, **kwargs):
        # reset seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # load
        self.file_tree.load(verbose=True)

        # build effects (such that their locations are constant across maha)
        self.effect_list = list()
        tqdm_dict = {'desc': 'sample effect mask',
                     'total': n_effect,
                     'disable': not verbose}
        for _ in tqdm(range(n_effect), **tqdm_dict):
            mask = Effect.sample_mask(prior_array=self.file_tree.mask,
                                      ref=self.file_tree.ref,
                                      **kwargs)

            # compute feat_stat in mask
            pc = PointCloud.from_mask(mask)
            fs = sum(self.file_tree.ijk_fs_dict[ijk] for ijk in pc)

            # maha below is placeholder, it is reset with each run
            e = Effect.from_fs_maha(fs=fs, mask=mask, maha=1)
            self.effect_list.append(e)

        # delete voxel wise stats
        self.file_tree.unload()

    def run_effect(self, effect, maha, active_rad=None, **kwargs):
        # get folder
        folder = increment_to_unique(self.folder / f'maha{maha:.3E}')

        # get mask of active area
        mask_active = self.file_tree.mask
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask_active = np.logical_and(mask_eff_dilated, mask_active)

        # set scale of effect
        effect.maha = maha

        # run effect
        run_arba(mask=mask_active,
                 effect=effect,
                 grp_effect=self.grp_effect,
                 folder_save=folder,
                 ft_dict=self.ft_dict,
                 **kwargs)

    def run(self, maha_list, par_flag=False, **kwargs):
        # build arg_list
        arg_list = list()
        for maha in maha_list:
            for effect in self.effect_list:
                d = {'effect': effect,
                     'maha': maha,
                     'verbose': not par_flag}
                d.update(kwargs)
                arg_list.append(d)

        # run
        if par_flag:
            parallel.run_par_fnc('run_effect', arg_list=arg_list, obj=self)
        if not par_flag:
            for d in arg_list:
                self.run_effect(**d)
