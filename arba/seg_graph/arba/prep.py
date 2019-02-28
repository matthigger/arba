import pathlib
import uuid

import numpy as np
from tqdm import tqdm

from mh_pytools import file
from ..file_tree import FileTree


def prep_arba(ft_dict, mask=None, grp_effect_dict=None, harmonize=False,
              verbose=False, folder=None, label=None, load_data=False,
              scale_equalize=True, par_flag=False, **kwargs):
    """ runs entire arba process, optionally saves outputs

    Args:
        ft_dict (dict): keys are population labels, values are FileTree
        mask (Mask): mask to operate in, if None defaults to all voxels which
                     are present in all images (see FileTree)
        grp_effect_dict (dict): keys are grp, values are effects to be applied
                                (defaults to no effects)
        harmonize (bool): toggles whether offset added to data so populations
                          have equal averages over active area ... otherwise
                          may 'discover' that the entire region is most sig
        verbose (bool): toggles command line output
        folder (str or Path): if passed, saves output.  otherwise no save
        label (str): if passed, printed to f_save to label output
        scale_equalize (bool): toggles whether scale is equalized (ensures
                               std dev is 1 for each feature internally). the
                               scale factors are kept in file_tree

    Returns:
        ft_dict (dict): same as input, now prepped
    """
    # get mask
    ft0, ft1 = tuple(ft_dict.values())
    if mask is None:
        if ft0.mask.ref != ft1.mask.ref:
            raise AttributeError('space mismatch between file trees')
        print('mask mismatch, taking intersection')
        mask = np.logical_and(ft0.mask, ft1.mask)
    ft0.mask = mask
    ft1.mask = mask

    # load data
    tqdm_dict = {'disable': not verbose,
                 'desc': 'load data, compute stats per voxel'}
    for ft in tqdm(ft_dict.values(), **tqdm_dict):
        ft.reset_hist()
        ft.load(verbose=verbose, load_data=load_data, par_flag=par_flag)

    # harmonize
    if harmonize:
        if verbose:
            print('harmonizing')
        harmonize_dict = FileTree.harmonize_via_add(ft_dict.values(),
                                                    apply=True,
                                                    verbose=verbose)

    # apply effects
    if grp_effect_dict is None:
        grp_effect_dict = dict()
    for grp, effect in grp_effect_dict.items():
        effect.apply_to_file_tree(ft_dict[grp])

    # scale equalize
    ft_tuple = tuple(ft_dict.values())
    if scale_equalize:
        FileTree.scale_equalize(ft_tuple, verbose=verbose, mask=mask)

    # save files
    if folder is not None:
        def print_feat_vec(x):
            s = ''
            for f in sorted(ft0.feat_list):
                idx = ft0.feat_list.index(f)
                s += f'{f} {x[idx]:.2e} '
            return s

        if label is None:
            label = str(uuid.uuid4())[:8]

        # save images and effects
        folder = pathlib.Path(folder)
        folder_save = folder / 'save'
        folder_save.mkdir(exist_ok=True, parents=True)

        mask.to_nii(folder / 'mask.nii.gz')

        file.save(ft_dict, folder_save / f'ft_dict_{label}.p.gz')

        for grp, effect in grp_effect_dict.items():
            effect.mask.to_nii(folder / f'mask_effect_{grp}.nii.gz')
            file.save(effect, folder_save / f'effect_{grp}.p.gz')

        # output mean images of each feature per grp (of testing dataset)
        feat_list = next(iter(ft_dict.values())).feat_list
        for feat in feat_list:
            for grp, ft in ft_dict.items():
                f_out = folder / f'{grp}_{feat}_{label}.nii.gz'
                ft.to_nii(f_out, feat=feat)

        # write description of prep
        f_save = folder / 'data_prep.txt'
        with open(str(f_save), 'a+') as f:
            if label is not None:
                print(f'{label}:', file=f)
            print(f'mask has {mask.sum()} voxels', file=f)
            print(f'number of images: ', file=f)
            for grp, ft in ft_dict.items():
                print(f'{len(ft):8.0f} {grp}', file=f)
            s_feat = ', '.join(sorted(ft.feat_list))
            print(f'features: {s_feat}', file=f)

            if harmonize:
                print('harmonization applied:' +
                      '\n    (offset added per grp so cross grp mean are equal)',
                      file=f)
                ft_dict_inv = {v: k for k, v in ft_dict.items()}
                _harmonize_dict = {ft_dict_inv[ft]: x
                                   for ft, x in harmonize_dict.items()}
                for grp in sorted(_harmonize_dict.keys()):
                    x = _harmonize_dict[grp]
                    print(f'    {grp} has offset: {print_feat_vec(x)}', file=f)
            else:
                print('harmonization not applied', file=f)
            for grp, effect in grp_effect_dict.items():
                np.set_printoptions(precision=3)
                n = effect.mask.sum()
                effect_mean = print_feat_vec(effect.mean)
                print(f'{grp} has effect applied: {effect_mean} @ {n} voxels',
                      file=f)
            print('', file=f)

    return ft_dict
