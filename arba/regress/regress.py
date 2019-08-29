import networkx as nx
import numpy as np
from scipy.spatial.distance import dice
from sortedcontainers.sortedset import SortedSet
from tqdm import tqdm

import arba
from mh_pytools import parallel
from bisect import bisect_right


def run_permute(feat_sbj, file_tree, fnc_target, save_folder=None,
                max_flag=True, fnc_tuple=None, n=0,
                reg_size_thresh=1, **kwargs):
    # ensure that fnc_target in fnc_tuple
    if fnc_tuple is None:
        fnc_tuple = fnc_target,
    elif fnc_target not in fnc_tuple:
        fnc_tuple = *fnc_tuple, fnc_target

    with file_tree.loaded():
        sg_hist = run_single(feat_sbj, file_tree, fnc_tuple=fnc_tuple,
                             **kwargs)

        # identify cutoff for signifigance
        if n == 0:
            val_list = []
            cutoff = np.inf
        else:
            val_list = permute(fnc_target, max_flag=max_flag,
                               feat_sbj=feat_sbj, file_tree=file_tree,
                               fnc_tuple=fnc_tuple, n=n,
                               reg_size_thresh=reg_size_thresh, **kwargs)

            # cutoff = np.percentile(val_list, cutoff_perc)

            #
            r2_null = np.vstack(val_list)
            r2_null = np.cumsum(r2_null, axis=1)
            r2_null = np.sort(r2_null, axis=0)

        num_perm = r2_null.shape[0]

        #
        merge_record = sg_hist.merge_record
        node_pval_dict = dict()
        node_fnc_dict = merge_record.fnc_node_val_list[fnc_target]
        for n in merge_record.nodes:
            node_size = merge_record.node_size_dict[n]
            if node_size < reg_size_thresh:
                continue

            # compute percentile
            pval = bisect_right(r2_null[:, node_size - 1], node_fnc_dict[n])
            pval /= num_perm
            if max_flag:
                pval = 1 - pval
            node_pval_dict[n] = pval

        if save_folder:
            merge_record.to_nii(save_folder / 'seg_hier.nii.gz', n=10)

            # # histogram of permutation testing
            # plt.hist(val_list, bins=30)
            # plt.axvline(cutoff, color='r',
            #             label=f'{cutoff_perc}th perc')
            # plt.legend()
            # plt.xlabel(f'max {fnc_target.__name__} per permutations')
            # plt.suptitle(f'{len(val_list)} permutations')
            # arba.plot.save_fig(save_folder / 'permute_r2.pdf')

    return sg_hist, node_pval_dict, val_list, r2_null


def build_mask(node_list, merge_record):
    # init empty mask
    ref = merge_record.ref
    assert ref.shape is not None, 'merge_record ref must have shape'
    mask = arba.space.Mask(np.zeros(ref.shape), ref=ref).astype(int)

    # sort nodes from biggest (value + space) to smallest
    node_set = SortedSet(node_list)
    while node_set:
        node = node_set.pop()

        # remove all the nodes which would be covered by node
        node_set -= set(nx.descendants(merge_record, node))

        # record position of node
        for ijk in merge_record.get_pc(node=node):
            mask[ijk] = node

    return mask


def run_single(feat_sbj, file_tree, fnc_tuple=None, permute_seed=None,
               **kwargs):
    if fnc_tuple is None:
        fnc_tuple = tuple()

    # set feat_sbj
    arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                           sbj_list=file_tree.sbj_list)

    if permute_seed:
        arba.region.RegionRegress.shuffle_feat_sbj(seed=permute_seed)

    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=file_tree,
                                             cls_reg=arba.region.RegionRegress,
                                             fnc_tuple=fnc_tuple)

    sg_hist.reduce_to(1, verbose=True, **kwargs)

    return sg_hist


def _get_val(fnc_target=None, max_flag=True, all_flag=False, reg_size_thresh=1,
             **kwargs):
    sg_hist = run_single(reg_size_thresh=reg_size_thresh, **kwargs)

    merge_record = sg_hist.merge_record
    node_val_dict = merge_record.fnc_node_val_list[fnc_target]
    val_list = [v for n, v in node_val_dict.items() if
                merge_record.node_size_dict[n] >= reg_size_thresh]

    if all_flag:
        val = sorted(val_list, reverse=True)
    elif max_flag:
        val = max(val_list)
    else:
        val = min(val_list)

    return val


def permute(fnc_target, feat_sbj, file_tree, n=5000, par_flag=False, **kwargs):
    arg_list = list()
    for n in range(n):
        d = {'feat_sbj': feat_sbj,
             'file_tree': file_tree,
             'permute_seed': n + 1,
             'fnc_target': fnc_target,
             'all_flag': True}
        d.update(**kwargs)
        arg_list.append(d)

    if par_flag:
        val_list = parallel.run_par_fnc(fnc=_get_val, arg_list=arg_list,
                                        verbose=True)
    else:
        val_list = list()
        for d in tqdm(arg_list, desc='permute'):
            val_list.append(_get_val(**d))

    # ensure that RegionRegress has appropriate feat_sbj ordering
    arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                           sbj_list=file_tree.sbj_list)

    return val_list


def compute_print_dice(mask_estimate, mask_target, save_folder):
    mask_estimate = mask_estimate.astype(bool)
    mask_target = mask_target.astype(bool)
    dice_score = 1 - dice(mask_estimate.flatten(), mask_target.flatten())
    with open(str(save_folder / 'dice.txt'), 'w') as f:
        print(f'dice is {dice_score:.3f}', file=f)
        print(f'target vox: {mask_target.sum()}', file=f)
        print(f'detected vox: {mask_estimate.sum()}', file=f)
        true_detect = (mask_target & mask_estimate).sum()
        print(f'true detected vox: {true_detect}', file=f)
        false_detect = (~mask_target & mask_estimate).sum()
        print(f'false detected vox: {false_detect}', file=f)

    return dice_score
