import pathlib

import numpy as np
from tqdm import tqdm

from mh_pytools import file
from .file_tree import FileTree
from .seg_graph_hist import SegGraphHistory


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
                          may 'discover' that the entire region is most sig
        verbose (bool): toggles command line output
        alpha (float): false positive rate

    Returns:
        sg_arba_test (SegGraph): candidate regions (test data)
    """
    if verbose:
        print('---begin seg graph init---')

    # get mask
    ft0, ft1 = tuple(ft_dict.values())
    if mask is None:
        assert np.allclose(ft0.mask, ft1.mask), 'mask mismatch'
        mask = ft0.mask
    else:
        ft0.mask = mask
        ft1.mask = mask

    # split into segmentation + test data
    ft_dict_seg = dict()
    ft_dict_test = dict()
    for grp, ft in ft_dict.items():
        ft_dict_seg[grp], ft_dict_test[grp] = ft.split()

    # todo: a lot of verbose code loading / harmonizing and applying ...
    # load data
    ft_list = list(ft_dict.values()) + \
              list(ft_dict_seg.values()) + \
              list(ft_dict_test.values())
    tqdm_dict = {'disable': not verbose,
                 'desc': 'load data, compute stats per voxel'}
    for ft in tqdm(ft_list, **tqdm_dict):
        ft.reset()
        ft.load(verbose=verbose, load_data=False)

    # harmonize
    if harmonize:
        tqdm_dict = {'disable': not verbose,
                     'desc': 'harmonizing per fold (segmentation + testing)'}
        ft_tuple_list = [tuple(ft_dict.values()),
                         tuple(ft_dict_test.values()),
                         tuple(ft_dict_seg.values())]
        for ft_tuple in tqdm(ft_tuple_list, **tqdm_dict):
            FileTree.harmonize_via_add(ft_tuple, apply=True, verbose=verbose)

    # apply effects
    if effect is not None:
        effect.apply_to_file_tree(ft_dict[grp_effect])
        effect.apply_to_file_tree(ft_dict_seg[grp_effect])
        effect.apply_to_file_tree(ft_dict_test[grp_effect])

    # build sg_hist_seg
    sg_hist_seg = SegGraphHistory(obj='maha', file_tree_dict=ft_dict_seg)

    if verbose:
        print('\n' * 3 + '---begin graph reduce---')

    # reduce
    sg_hist_seg.reduce_to(1, verbose=verbose, **kwargs)

    if verbose:
        print('\n' * 3 + '---begin saving output---')

    # save
    if folder_save is not None:
        file.save(sg_hist_seg, folder_save / 'sg_hist_seg.p.gz')

    # determine candidate regions
    sg_arba_seg = sg_hist_seg.cut_greedy_sig(alpha=alpha)

    # swap data source for test data
    sg_arba_test = sg_arba_seg.from_file_tree_dict(ft_dict_test)
    sg_hist_test = sg_hist_seg.from_file_tree_dict(ft_dict_test)

    #
    sg_arba_test_sig = sg_arba_test.get_sig(alpha=alpha)

    # save
    if folder_save is not None:
        folder_save = pathlib.Path(folder_save)
        folder_save_image = folder_save / 'image'
        folder_save_image.mkdir()

        file.save(ft_dict, folder_save / 'ft_dict.p.gz')
        file.save(ft_dict_seg, folder_save / 'ft_dict_seg.p.gz')
        file.save(ft_dict_test, folder_save / 'ft_dict_test.p.gz')
        file.save(sg_arba_seg, folder_save / 'sg_arba_seg.p.gz')
        file.save(sg_arba_test, folder_save / 'sg_arba_test.p.gz')
        file.save(sg_hist_test, folder_save / 'sg_hist_test.p.gz')
        file.save(sg_arba_test_sig, folder_save / 'sg_arba_test_sig.p.gz')

        # save mask
        mask.to_nii(folder_save_image / 'mask.nii.gz')
        sg_arba_test_sig.to_nii(folder_save_image / 'mask_sig_arba.nii.gz',
                                ref=mask.ref,
                                fnc=lambda r: 1,
                                background=0)

        # output mean images of each feature per grp (of testing dataset)
        feat_list = next(iter(ft_dict.values())).feat_list
        for feat in feat_list:
            for grp, ft in ft_dict_test.items():
                f_out = folder_save_image / f'{grp}_{feat}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # save effect
        if effect is not None:
            effect.mask.to_nii(folder_save_image / f'mask_effect.nii.gz')
            file.save(effect, folder_save / f'effect.p.gz')

    return sg_arba_test
