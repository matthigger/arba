import pathlib

import numpy as np
from tqdm import tqdm

from mh_pytools import file
from .data import FileTree
from .factory import seg_graph_factory


def run_arba(ft_dict, mask=None, folder_save=None, effect=None,
             grp_effect=None, harmonize=True, verbose=False, alpha=.05,
             **kwargs):
    """ runs entire arba process, optionally saves outputs

    Args:
        ft_dict (dict): keys are population labels, values are FileTree
        mask (Mask): mask to operate in, if None defaults to all voxels which
                     are present in all images (see FileTree)
        folder_save (str or Path): output folder for experiment, if None will
                                   not save
        effect (Effect): effect to apply
        grp_effect : target group of which effect is applied to
        harmonize (bool): toggles whether offset added to data so populations
                          have equal averages over active area ... otherwise
                          may 'discover' that the entire region is most sig.
                          note: only applied to segmentation data set
        verbose (bool): toggles command line output
        alpha (float): false positive rate

    Returns:
        sg_arba_test (SegGraph): candidate regions (test data)
    """
    # get mask
    if mask is None:
        m0, m1 = tuple(ft.get_mask() for ft in ft_dict.values())
        mask = np.logical_and(m0, m1)

    # split into segmentation + test data, load it
    ft_dict_seg = dict()
    ft_dict_test = dict()
    tqdm_dict = {'disable': not verbose, 'desc': 'loading data per grp'}
    for grp, ft in tqdm(ft_dict.items(), **tqdm_dict):
        ft_dict_seg[grp], ft_dict_test[grp] = ft.split()
        ft_dict_seg[grp].load(mask=mask, verbose=verbose)
        ft_dict_test[grp].load(mask=mask, verbose=verbose)

    # harmonize
    if harmonize:
        FileTree.harmonize_via_add(ft_dict_seg.values(), apply=True)
        FileTree.harmonize_via_add(ft_dict_test.values(), apply=True)

    # apply effects
    if effect is not None:
        effect.apply_to_file_tree(ft_dict_seg[grp_effect])
        effect.apply_to_file_tree(ft_dict_test[grp_effect])

    # build sg_hist
    sg_hist = seg_graph_factory(obj='maha', file_tree_dict=ft_dict_seg,
                                history=True, **kwargs)
    sg_hist.reduce_to(1, verbose=True)

    # determine candidate regions
    sg_arba = sg_hist.cut_greedy_pval(alpha=alpha)

    # swap data source for test data
    sg_arba_test = sg_arba.from_file_tree_dict(ft_dict_test)
    sg_hist_test = sg_hist.from_file_tree_dict(ft_dict_test)

    #
    sg_arba_test_sig = sg_arba_test.is_sig(alpha=alpha)

    # save
    if folder_save is not None:
        folder_save = pathlib.Path(folder_save)
        file.save(ft_dict, folder_save / 'ft_dict.p.gz')
        file.save(ft_dict_seg, folder_save / 'ft_dict_seg.p.gz')
        file.save(ft_dict_test, folder_save / 'ft_dict_test.p.gz')
        file.save(sg_hist, folder_save / 'sg_hist.p.gz')
        file.save(sg_arba, folder_save / 'sg_arba.p.gz')
        file.save(sg_arba_test, folder_save / 'sg_arba_test.p.gz')
        file.save(sg_hist_test, folder_save / 'sg_hist_test.p.gz')
        file.save(sg_arba_test_sig, folder_save / 'sg_arba_test_sig.p.gz')

        # save mask
        mask.to_nii(folder_save / 'mask.nii.gz')
        sg_arba_test_sig.to_nii(folder_save / 'mask_sig_arba.nii.gz',
                                ref=mask.ref,
                                fnc=lambda r: 1,
                                background=0)

        # output mean images of each feature per grp (of testing dataset)
        feat_list = next(iter(ft_dict.values())).feat_list
        for feat in feat_list:
            for grp, ft in ft_dict_test.items():
                f_out = folder_save / f'{grp}_{feat}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # save effect
        if effect is not None:
            effect.mask.to_nii(folder_save / f'mask_effect.nii.gz')
            file.save(effect, folder_save / f'effect.p.gz')

    return sg_arba_test
