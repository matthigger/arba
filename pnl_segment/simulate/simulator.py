import pathlib

import numpy as np

from .effect import Effect
from ..seg_graph import FeatStatEmpty, FileTree, run_arba
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
    """ manages simulation: paths, samples effects, runs seg_graph.reduce()

    Attributes:
        file_tree (FileTree):
    """
    grp_effect = 'grp_effect'
    grp_null = 'grp_null'

    def __init__(self, file_tree, folder, eff_prior_arr=None, p_effect=.5):
        self.ref = file_tree.ref
        self.pc = PointCloud.from_mask(file_tree.mask)
        self.feat_list = file_tree.feat_list
        self.folder = folder

        self.eff_prior_arr = eff_prior_arr
        if self.eff_prior_arr is None:
            self.eff_prior_arr = file_tree.mask

        # split into two file_trees
        ft_eff, ft_null = file_tree.split(p=p_effect)
        self.ft_dict = {self.grp_effect: ft_eff,
                        self.grp_null: ft_null}

    def run_effect(self, maha, active_rad=None, effect_mask=None,
                   harmonize=True, **kwargs):
        # get folder
        folder = increment_to_unique(self.folder / f'maha{maha:.3E}')

        # sample effect
        if effect_mask is None:
            effect_mask = Effect.sample_mask(prior_array=self.eff_prior_arr)

        # get mask of active area
        mask_active = self.pc.to_mask()
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect_mask.dilate(active_rad)
            mask_active = np.logical_and(mask_eff_dilated, mask_active)

        # (ft_dict has no memory intersection with self.ft_dict)
        ft_dict = dict()
        for grp, ft in self.ft_dict.items():
            ft_dict[grp] = FileTree(sbj_feat_file_tree=ft.sbj_feat_file_tree)
            ft_dict[grp].load(mask=mask_active, **kwargs)

        # compute stats of all observed data within mask
        effect_pc = PointCloud.from_mask(effect_mask)
        fs = FeatStatEmpty()
        for ft in ft_dict.values():
            for ijk in effect_pc:
                fs += ft.ijk_fs_dict[ijk]

        # build effect
        effect = Effect.from_data(fs=fs, maha=maha, mask=effect_mask, **kwargs)

        # run effect
        run_arba(ft_dict=ft_dict,
                 mask=mask_active,
                 effect=effect,
                 grp_effect=self.grp_effect,
                 folder_save=folder,
                 harmonize=harmonize)
