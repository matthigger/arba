import pathlib
import random

from mh_pytools import file
from .prep import prep_arba
from ..seg_graph_hist import SegGraphHistory
from ..seg_graph_perm import SegGraphPerm


def run_arba_permute(ft_dict, folder_save=None, verbose=False, alpha=.05,
                     par_flag=False, seed=1, n_permute=100, **kwargs):
    """ runs arba (cross validation), optionally saves outputs.

    Args:
        ft_dict (dict): keys are population labels, values are FileTree
        folder_save (str or Path): output folder for experiment, if None will
                                   not save
        verbose (bool): toggles command line output
        alpha (float): false positive rate
        par_flag (bool): toggles parallel computation
        seed : seed to initialize permutation testing with
        n_permute (int): number of permutations

    Returns:
        sg_arba_test (SegGraph): candidate regions (test data)
    """
    # prep
    ft_dict = prep_arba(ft_dict, label='full', folder_save=folder_save,
                        load_data=True, **kwargs)

    # build sg_hist
    sg_hist = SegGraphHistory(obj='maha', file_tree_dict=ft_dict)

    # reduce
    sg_hist.reduce_to(1, verbose=verbose, **kwargs)

    # permutation testing
    # aggregate data
    ft0, ft1 = ft_dict.values()
    ft = ft0 | ft1

    # determine splits
    n0 = len(ft0)
    random.seed(seed)
    arg_list = list()
    for _ in range(n_permute):
        d = {'ft': ft,
             'sbj_set0': set(random.sample(ft.sbj_list, k=n0))}
        d.update(kwargs)
        arg_list.append(d)

    # run each permutation of the data
    if par_flag:
        raise NotImplementedError
    else:
        max_maha_list = list()
        for d in arg_list:
            max_maha_list.append(_run_permute(**d))

    # determine which regions are sig
    raise NotImplementedError
    sg_arba_sig = 3

    # save
    if folder_save is not None:
        folder_save = pathlib.Path(folder_save)
        folder_save_image = folder_save / 'image'
        folder_save_image.mkdir(exist_ok=True)

        file.save(ft_dict, folder_save / 'ft_dict.p.gz')

        file.save(sg_hist, folder_save / 'sg_hist.p.gz')
        file.save(sg_arba_test_sig, folder_save / 'sg_arba_test_sig.p.gz')

        # save sig mask
        ref = next(iter(ft_dict.values())).ref
        sg_arba_test_sig.to_nii(folder_save_image / 'mask_sig_arba.nii.gz',
                                ref=ref,
                                fnc=lambda r: 1,
                                background=0)

    return sg_arba_sig


def _run_permute(ft, sbj_set0, **kwargs):
    """ runs """
    ft_dict = dict(enumerate(ft.split(sbj_set0=sbj_set0)))

    # build sg_perm
    sg_perm = SegGraphPerm(obj='maha', file_tree_dict=ft_dict)

    # reduce
    sg_perm.reduce_to(1, verbose=False, **kwargs)

    return sg_perm.max_maha
