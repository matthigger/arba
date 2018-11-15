import pathlib

import numpy as np

from mh_pytools import file
from .effect import Effect, EffectDm
from ..region import RegionMaha
from ..seg_graph import seg_graph_factory, FeatStatEmpty
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
        self.pc = file_tree.get_point_cloud()
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

    def run(self, effect, obj, folder, active_rad=None, verbose=False,
            f_rba=None, resample=False, **kwargs):
        """ runs experiment
        """
        # get mask of active area
        mask = self.pc.to_mask()
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask = np.logical_and(mask_eff_dilated, mask)

        # apply mask (and copies file_tree, ft_dict has no memory intersection
        # with self.ft_dict)
        ft_dict = {label: ft.apply_mask(mask)
                   for label, ft in self.ft_dict.items()}

        # resample if need be:
        if resample:
            for grp, ft in ft_dict.items():
                ft.resample_iid(effect.mask)

        # apply effect to effect group
        ft_effect = ft_dict[self.grp_effect]
        ft_dict[self.grp_effect] = effect.apply_to_file_tree(ft_effect)

        # output mean images of each feature per grp
        for feat in self.feat_list:
            for grp, ft in ft_dict.items():
                f_out = folder / f'{grp}_{feat}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # save effect
        if not isinstance(effect, EffectDm):
            f_mask_effect = folder / 'effect_mask.nii.gz'
            effect.mask.to_nii(f_mask_effect)
            file.save(effect, file=folder / 'effect.p.gz')

        # build part seg_graph
        sg_hist = seg_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                    history=True)

        def weighted_maha(reg):
            return RegionMaha.get_obj(reg) * len(reg)

        # save mask
        mask.to_nii(folder / 'active_mask.nii.gz')
        file.save(sg_hist, file=folder / 'sg_vba.p.gz')
        sg_hist.to_nii(f_out=folder / f'{obj}_vba.nii.gz',
                       ref=self.ref,
                       fnc=weighted_maha)

        # reduce + save
        sg_hist.reduce_to(1, verbose=verbose)
        file.save(sg_hist, file=folder / 'sg_hist.p.gz')

        # split tree into regions + save
        sg_span = sg_hist.get_min_error_span()
        file.save(sg_span, file=folder / 'sg_arba.p.gz')

        sg_span.to_nii(f_out=folder / f'{obj}_arba.nii.gz',
                       ref=self.ref,
                       fnc=weighted_maha)

        if f_rba is not None:
            # build naive segmentation, aggregate via a priori atlas
            sg_rba = seg_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                       history=False)
            sg_rba.combine_by_reg(f_rba)
            file.save(sg_rba, file=folder / 'sg_rba.p.gz')
            sg_rba.to_nii(f_out=folder / f'{obj}_rba.nii.gz',
                          ref=self.ref,
                          fnc=weighted_maha)

            if not isinstance(effect, EffectDm):
                # build 'perfect' segmentation, aggregate only effect mask
                sg_rba = seg_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                           history=False)
                sg_rba.combine_by_reg(f_mask_effect)
                file.save(sg_rba, file=folder / 'sg_truth.p.gz')
                sg_rba.to_nii(f_out=folder / f'{obj}_truth.nii.gz',
                              ref=self.ref,
                              fnc=weighted_maha)

    def run_healthy(self, obj, **kwargs):
        eff = EffectDm()
        folder = increment_to_unique(self.folder / 'healthy')
        self.run(effect=eff, obj=obj, folder=folder, **kwargs)

    def run_effect(self, snr, obj, **kwargs):
        eff = self.sample_effect(snr, **kwargs)
        folder = increment_to_unique(self.folder / f'snr_{snr:.3E}')
        self.run(effect=eff, obj=obj, folder=folder, **kwargs)
