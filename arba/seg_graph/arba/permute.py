import pathlib
import random
from bisect import bisect_right

from tqdm import tqdm

from mh_pytools import file
from mh_pytools.parallel import run_par_fnc
from .prep import prep_arba
from ..seg_graph import SegGraph
from ..seg_graph_maha import SegGraphMaha


def run_arba_permute(ft_dict, folder=None, verbose=False, alpha=.05,
                     par_flag=True, seed=1, n_permute=100, **kwargs):
    """ runs arba (cross validation), optionally saves outputs.

    Args:
        ft_dict (dict): keys are population labels, values are FileTree
        folder (str or Path): output folder for experiment, if None will
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
    ft_dict = prep_arba(ft_dict, label='full', folder=folder,
                        load_data=True, **kwargs)

    # build sg_hist
    sg_hist = SegGraphMaha(file_tree_dict=ft_dict)

    # reduce
    sg_hist.reduce_to(1, verbose=verbose, **kwargs)

    # permutation testing
    # aggregate data
    ft0, ft1 = ft_dict.values()
    ft = ft0 | ft1

    # determine splits
    n0 = len(ft0)
    random.seed(seed)
    desc = 'splitting data'
    if par_flag:
        _arg_list = [{'ft': ft, 'n': n0}] * n_permute
        arg_list = run_par_fnc(_split, _arg_list, desc=desc, verbose=verbose)
    else:
        tqdm_dict = {'disable': not verbose,
                     'desc': desc}
        arg_list = list()
        for _ in tqdm(range(n_permute), **tqdm_dict):
            arg_list.append(_split(ft, n0))

    # run each permutation of the data
    if par_flag:
        max_maha_list = run_par_fnc(_run_permute, arg_list, desc='permutation')
    else:
        max_maha_list = list()
        for d in arg_list:
            max_maha_list.append(_run_permute(**d))

    # determine which regions are sig
    sg_arba_max_maha = sg_hist.cut_greedy_maha()
    sg_arba_sig = get_sig(sg_arba_max_maha, max_maha_list, alpha=alpha)

    # save
    if folder is not None:
        folder = pathlib.Path(folder)
        folder_save = folder / 'save'
        folder_save.mkdir(exist_ok=True)

        file.save(ft_dict, folder_save / 'ft_dict.p.gz')

        file.save(sg_hist, folder_save / 'sg_hist.p.gz')
        file.save(sg_arba_max_maha, folder_save / 'sg_arba_max_maha.p.gz')
        file.save(sg_arba_sig, folder_save / 'sg_arba_sig.p.gz')

        # save sig mask
        ref = next(iter(ft_dict.values())).ref
        sg_arba_sig.to_nii(folder / 'mask_sig_arba_permute.nii.gz',
                           ref=ref,
                           fnc=lambda r: 1,
                           background=0)

    return sg_arba_sig


def _run_permute(ft_dict, **kwargs):
    """ runs """
    # build sg_maha
    sg_maha = SegGraphMaha(file_tree_dict=ft_dict)

    # reduce
    sg_maha.reduce_to(1, verbose=False, **kwargs)

    return max(sg_maha.node_maha_dict.values())


def _split(ft, n):
    # split data
    sbj_set0 = set(random.sample(ft.sbj_list, k=n))
    ft_tuple = ft.split(sbj_set0=sbj_set0)

    # rm raw data to save memory (not needed)
    for _ft in ft_tuple:
        _ft.data = None

    # build ft_dict, update arg_list
    d = {'ft_dict': dict(enumerate(ft_tuple))}
    return d


def get_sig(sg, max_maha_list, alpha=.05):
    """ builds copy of sg containing only significance regions

    # todo: compute confidence intervals using fnc below:
    # from statsmodels.stats.proportion import proportion_confint

    Args:
        sg (SegGraph): a seg graph
        max_maha_list (list): list of bootstrapped max mahalanobis distances
        alpha (float): false positive rate

    Returns:
        sg_sig (SegGraph): a seg graph containing only sig regions
    """

    # prep
    max_maha_list = sorted(max_maha_list)
    n = len(max_maha_list)

    # determine which regions are sig
    reg_list = list()
    for reg in sg.nodes:
        # k is the number of permutations for which observed maha is >= to
        k = bisect_right(max_maha_list, reg.maha * len(reg))
        if (k / n) >= (1 - alpha):
            reg_list.append(reg)

    # build seg graph
    sg_sig = SegGraph(obj=sg.reg_type, file_tree_dict=sg.file_tree_dict,
                      _add_nodes=False)
    sg_sig.add_nodes_from(reg_list)

    return sg_sig
