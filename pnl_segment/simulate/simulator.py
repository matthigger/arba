import pathlib

import numpy as np

from mh_pytools import file
from .effect import Effect
from ..region import RegionMaha
from ..seg_graph import seg_graph_factory, FeatStatEmpty, FileTree
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
    grp_effect = 'effect'
    grp_null = 'null'

    def __init__(self, file_tree, folder, eff_prior_arr=None, p_effect=.5,
                 **kwargs):
        self.ref = file_tree.ref
        self.pc = file_tree.point_cloud
        self.feat_list = file_tree.feat_list
        self.folder = folder

        self.eff_prior_arr = eff_prior_arr
        if self.eff_prior_arr is None:
            self.eff_prior_arr = self.pc.to_mask(ref=file_tree.ref)

        # split into two file_trees
        ft_eff, ft_null = file_tree.split(p=p_effect, **kwargs)
        self.ft_dict = {self.grp_effect: ft_eff,
                        self.grp_null: ft_null}

    def sample_effect_mask(self, radius):
        effect_mask = Effect.sample_mask(prior_array=self.eff_prior_arr,
                                         radius=radius)
        effect_mask.ref = self.ref

        return effect_mask

    def sample_effect(self, snr, effect_mask=None, **kwargs):
        """ generates effect at random location with given snr to healthy
        """
        if effect_mask is None:
            effect_mask = self.sample_effect_mask(**kwargs)

        # find intersection of effect mask and observations
        pc = self.pc.intersection(PointCloud.from_mask(effect_mask))

        # compute stats of all observed data within mask
        fs = FeatStatEmpty()
        for ft in self.ft_dict.values():
            for ijk in pc:
                fs += ft.ijk_fs_dict[ijk]

        # build effect
        effect = Effect.from_data(fs=fs, snr=snr, mask=effect_mask, **kwargs)

        return effect

    def run(self, effect, obj, ft_dict, verbose=False, resample=False,
            save=False, harmonize=False, **kwargs):
        """ runs experiment
        """

        # resample if need be:
        if resample and np.count_nonzero(effect.mask):
            for grp, ft in ft_dict.items():
                ft.resample_iid(effect.mask)

        # harmonize
        if harmonize:
            mu_offset = FileTree.harmonize_via_add(ft_dict.values(),
                                                   apply=True)

        # apply effect to effect group
        ft_effect = ft_dict[self.grp_effect]
        ft_dict[self.grp_effect] = effect.apply_to_file_tree(ft_effect)

        # build part seg_graph
        sg_hist = seg_graph_factory(obj=obj,
                                    file_tree_dict=ft_dict,
                                    history=True)

        # reduce
        sg_hist.reduce_to(1, verbose=verbose)

        # save
        if save:
            self.save(sg_hist, ft_dict, effect, obj, **kwargs)

        return sg_hist

    def save(self, sg_hist, ft_dict, effect, obj, folder, f_rba=None,
             **kwargs):

        # save sg_hist
        file.save(sg_hist, file=folder / 'sg_hist.p.gz')

        # output mean images of each feature per grp
        for feat in self.feat_list:
            for grp, ft in ft_dict.items():
                f_out = folder / f'{grp}_{feat}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # save effect
        f_mask_effect = folder / 'effect_mask.nii.gz'
        effect.mask.to_nii(f_mask_effect)
        file.save(effect, file=folder / 'effect.p.gz')

        # build sg_vba / sg_rba
        sg_vba = seg_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                   history=False)
        self.save_sg(sg=sg_vba, folder=folder, label='vba')

        if f_rba is not None:
            sg_vba.combine_by_reg(f_rba)
            self.save_sg(sg=sg_vba, folder=folder, label='rba')

    def save_sg(self, sg, label, folder):
        def get_maha(reg):
            return RegionMaha.get_obj(reg)

        def get_wmaha(reg):
            return RegionMaha.get_obj(reg) * len(reg)

        file.save(sg, file=folder / f'sg_{label}.p.gz')
        sg.to_nii(f_out=folder / f'maha_{label}.nii.gz',
                  ref=self.ref,
                  fnc=get_maha)
        sg.to_nii(f_out=folder / f'wmaha_{label}.nii.gz',
                  ref=self.ref,
                  fnc=get_wmaha)

    def run_effect(self, snr, obj, active_rad=None, **kwargs):
        # get folder
        folder = increment_to_unique(self.folder / f'snr_{snr:.3E}')

        # sample effect
        effect = self.sample_effect(snr, **kwargs)

        # get mask of active area + save
        mask_active = self.pc.to_mask(self.ref)
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask_active = np.logical_and(mask_eff_dilated, mask_active)
        mask_active.to_nii(folder / 'active_mask.nii.gz')

        # apply mask (and copies file_tree, ft_dict has no memory intersection
        # with self.ft_dict)
        ft_dict = {label: ft.apply_mask(mask_active)
                   for label, ft in self.ft_dict.items()}

        # # run sham
        # ft0, ft1 = ft_dict[self.grp_null].split()
        # ft_dict_dm = {self.grp_null: ft0, self.grp_effect: ft1}
        # sg_hist_dm = self.run(effect=effect.make_dm_effect(),
        #                       obj=obj,
        #                       ft_dict=ft_dict_dm,
        #                       **kwargs)
        # # maha_scaler = sg_hist_dm.get_maha_scaler()
        # file.save(sg_hist_dm, folder / 'sg_hist_dm.p.gz')

        # run effect
        sg_hist = self.run(effect=effect,
                           obj=obj,
                           folder=folder,
                           ft_dict=ft_dict,
                           save=True,
                           **kwargs)
