import pathlib
import random

import numpy as np
from tqdm import tqdm

from mh_pytools import parallel
from . import Effect
from ..seg_graph import run_arba_cv, run_arba_permute
from ..space import PointCloud


class Simulator:
    """
    Attributes:
        folder (Path): location of output
        file_tree (FileTree): full file tree of all healthy sbj
        p_effect (float): percentage of sbj which have effect applied
        effect_list (list): list of effects to apply
        modes (tuple): includes 'cv' and 'permute'
    """

    grp_effect = 'grp_effect'
    grp_null = 'grp_null'
    mode_fnc_name_dict = {'cv': 'run_effect_cv',
                          'permute': 'run_effect_permute'}

    def __init__(self, folder, file_tree, p_effect=.5,
                 modes=('permute', 'cv')):
        self.folder = pathlib.Path(folder)
        folder.mkdir(parents=True)

        # split into two file_trees
        self.file_tree = file_tree
        ft_eff, ft_null = file_tree.split(p=p_effect)
        self.ft_dict = {self.grp_effect: ft_eff,
                        self.grp_null: ft_null}

        self.effect_list = list()
        self.modes = modes

    def build_effect_list(self, radius=None, num_vox=None, verbose=False,
                          seed=1, seg_array=None):
        # reset seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # load
        self.file_tree.load(verbose=True)

        # build mask list (corresponds to effect location)
        mask_list = list()
        tqdm_dict = {'desc': 'sample effect mask',
                     'disable': not verbose}
        if (radius is None) == (num_vox is None):
            raise AttributeError('either radius xor num_vox required')
        elif radius is not None:
            for rad in tqdm(radius, **tqdm_dict):
                mask = Effect.sample_mask(prior_array=self.file_tree.mask,
                                          ref=self.file_tree.ref,
                                          radius=rad,
                                          seg_array=seg_array)
                mask_list.append(mask)
        else:
            for n in tqdm(num_vox, **tqdm_dict):
                mask = Effect.sample_mask(prior_array=self.file_tree.mask,
                                          ref=self.file_tree.ref,
                                          num_vox=n,
                                          seg_array=seg_array)
                mask_list.append(mask)

        # build effects (such that their locations are constant across maha)
        self.effect_list = list()
        for mask in mask_list:
            # compute feat_stat in mask
            pc = PointCloud.from_mask(mask)
            fs = sum(self.file_tree.ijk_fs_dict[ijk] for ijk in pc)

            # maha below is placeholder, it is reset with each run
            e = Effect.from_fs_maha(fs=fs, mask=mask, maha=1)
            self.effect_list.append(e)

        # delete voxel wise stats
        self.file_tree.unload()

    def run_effect_prep(self, effect, maha=None, active_rad=None, **kwargs):
        # get mask of active area
        mask_active = self.file_tree.mask
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask_active = np.logical_and(mask_eff_dilated, mask_active)

        # set scale of effect
        if maha is not None:
            effect.maha = maha

        # build effect dict
        grp_effect_dict = {self.grp_effect: effect}

        return mask_active, grp_effect_dict

    def run_effect_permute(self, effect, folder, par_flag=False, **kwargs):
        mask, grp_effect_dict = self.run_effect_prep(effect, **kwargs)

        run_arba_permute(mask=mask,
                         grp_effect_dict=grp_effect_dict,
                         folder=folder / 'arba_permute',
                         ft_dict=self.ft_dict,
                         par_flag=par_flag,
                         **kwargs)

    def run_effect_cv(self, effect, folder, **kwargs):
        mask, grp_effect_dict = self.run_effect_prep(effect, **kwargs)

        run_arba_cv(mask=mask,
                    grp_effect_dict=grp_effect_dict,
                    folder=folder / 'arba_cv',
                    ft_dict=self.ft_dict,
                    **kwargs)

    def run(self, maha_list, par_flag=False, par_permute_flag=False,
            **kwargs):

        if par_flag and par_permute_flag:
            raise AttributeError('par_flag or par_permute_flag must be false')

        # build arg_list
        arg_list = list()
        z_width_maha = np.ceil(np.log10(len(maha_list))).astype(int)
        z_width_eff = np.ceil(np.log10(len(self.effect_list))).astype(int)
        for maha_idx, maha in enumerate(sorted(maha_list)):
            for eff_idx, effect in enumerate(self.effect_list):
                s_maha = str(maha_idx).zfill(z_width_maha)
                s_eff = str(eff_idx).zfill(z_width_eff)
                folder = self.folder / f'maha{s_maha}_{maha:.1E}_effect{s_eff}'

                d = {'effect': effect,
                     'maha': maha,
                     'verbose': not par_flag,
                     'par_flag': par_permute_flag,
                     'folder': folder}
                d.update(kwargs)
                arg_list.append(d)

        # run
        for mode in self.modes:
            fnc_name = Simulator.mode_fnc_name_dict[mode]
            desc = f'simulating effects ({mode})'

            if par_flag:
                parallel.run_par_fnc(fnc_name, arg_list=arg_list, obj=self,
                                     desc=desc)
            if not par_flag:
                for d in arg_list:
                    fnc = getattr(self, fnc_name)
                    fnc(**d)
