import pathlib

import numpy as np

from mh_pytools import file
from pnl_segment.adaptive.part_graph_factory import part_graph_factory
from pnl_segment.simulate.effect import Effect, EffectDm
from pnl_segment.simulate.mask import Mask


def increment_to_unique(folder, num_width=3):
    idx = 0
    while True:
        _folder = pathlib.Path(
            str(folder) + '_run' + str(idx).zfill(num_width))
        if not _folder.exists():
            return _folder
        idx += 1


class Simulator:
    """ manages simulation: paths, samples effects, runs part_graph.reduce()

    Attributes:
        file_tree (FileTree):
    """

    def __init__(self, file_tree, folder, eff_prior_arr=None):
        self.file_tree = file_tree
        self.ft_dict = None
        self.folder = folder

        self.eff_prior_arr = eff_prior_arr
        if self.eff_prior_arr is None:
            self.eff_prior_arr = np.zeros(shape=file_tree.ref.shape)
            for ijk in file_tree.ijk_fs_dict.keys():
                self.eff_prior_arr[ijk] = 1

    def split(self, p_effect=.5, lossy=True):
        if self.file_tree is None:
            raise AttributeError('simulator has previously been split')

        # split into two file_tree
        grp_list = ('effect', 'null')
        ft_tuple = self.file_tree.split(p=p_effect,
                                        unload_self=lossy,
                                        unload_kids=True)
        self.ft_dict = dict(zip(grp_list, ft_tuple))

    def sample_effect(self, snr, radius=3, **kwargs):
        """ generates effect at random location with given snr to healthy
        """
        effect_mask = Effect.sample_mask(prior_array=self.eff_prior_arr,
                                         radius=radius)
        effect_mask.ref_space = self.file_tree.ref

        effect = Effect.from_data(ijk_fs_dict=self.file_tree.ijk_fs_dict,
                                  mask=effect_mask,
                                  snr=snr)

        return effect

    def run(self, effect, obj, folder, active_rad=None, verbose=False,
            f_rba=None, **kwargs):
        """ runs experiment
        """
        # get mask of active area
        mask = self.file_tree.get_mask()
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask = Mask.build_intersection([mask_eff_dilated, mask])

        # apply mask
        ft_dict = {label: ft.apply_mask(mask)
                   for label, ft in self.ft_dict.items()}

        # apply effect to effect group
        for label, ft in ft_dict.items():
            ft_dict[label] = effect.apply_to_file_tree(ft)

        # output mean images of each feature per grp
        folder.mkdir(exist_ok=True, parents=True)
        for feat in self.file_tree.feat_list:
            for grp, ft in ft_dict.items():
                f_out = folder / f'{grp}_{feat}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # save effect
        if not isinstance(effect, EffectDm):
            effect.mask.to_nii(folder / 'effect_mask.nii.gz')
            file.save(effect, file=folder / 'effect.p.gz')

        # build part graph
        pg_hist = part_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                     history=True)

        def get_obj(reg):
            return -reg.obj

        file.save(pg_hist, file=folder / 'pg_init.p.gz')
        pg_hist.to_nii(f_out=folder / f'{obj}_vba.nii.gz',
                       ref=self.file_tree.ref,
                       fnc=get_obj)

        # reduce
        pg_hist.reduce_to(1, verbose=verbose)

        # build arba segmentation
        def spanning_fnc(reg):
            return reg.obj

        pg_span = pg_hist.get_min_spanning_region(spanning_fnc)

        # save
        file.save(pg_span, file=folder / 'pg_span.p.gz')

        file.save(pg_hist, file=folder / 'pg_hist.p.gz')

        pg_span.to_nii(f_out=folder / f'{obj}_arba.nii.gz',
                       ref=self.file_tree.ref,
                       fnc=get_obj)

        if f_rba is not None:
            pg_rba = part_graph_factory(obj=obj, file_tree_dict=ft_dict,
                                        history=False)
            pg_rba.combine_by_reg(f_rba)
            file.save(pg_rba, file=folder / 'pg_rba.p.gz')
            pg_rba.to_nii(f_out=folder / f'{obj}_rba.nii.gz',
                          ref=self.file_tree.ref,
                          fnc=get_obj)

        return pg_span, pg_hist

    def run_healthy(self, obj, **kwargs):
        eff = EffectDm()
        folder = increment_to_unique(self.folder / 'healthy')
        self.run(effect=eff, obj=obj, folder=folder, **kwargs)

    def run_effect(self, snr, obj, **kwargs):
        eff = self.sample_effect(snr, **kwargs)
        folder = increment_to_unique(self.folder / f'snr_{snr:.3E}')
        self.run(effect=eff, obj=obj, folder=folder, **kwargs)
