import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import dice
from sortedcontainers.sortedset import SortedSet
from tqdm import tqdm

import arba


def run_permute(feat_sbj, file_tree, fnc_target, save_folder=None,
                max_flag=True, cutoff_perc=95, fnc_tuple=None,
                resolve_flag=False, **kwargs):
    # ensure that fnc_target in fnc_tuple
    if fnc_tuple is None:
        fnc_tuple = fnc_target,
    elif fnc_target not in fnc_tuple:
        fnc_tuple = *fnc_tuple, fnc_target

    if not max_flag:
        # in min mode, we need 'inverse' of cutoff_perc
        cutoff_perc = 100 - cutoff_perc

    with file_tree.loaded():
        sg_hist = run_single(feat_sbj, file_tree, fnc_tuple=fnc_tuple,
                             **kwargs)

        # identify cutoff for signifigance
        val_list = permute(fnc_target, max_flag=max_flag,
                           feat_sbj=feat_sbj, file_tree=file_tree,
                           fnc_tuple=fnc_tuple, **kwargs)
        cutoff = np.percentile(val_list, cutoff_perc)

        #
        merge_record = sg_hist.merge_record
        sig_node_list = list()
        node_fnc_dict = merge_record.fnc_node_val_list[fnc_target]
        for n in merge_record.nodes:
            if (not max_flag and node_fnc_dict[n] <= cutoff) or \
                    (max_flag and node_fnc_dict[n] >= cutoff):
                sig_node_list.append(n)

        if save_folder:
            merge_record.to_nii(save_folder / 'seg_hier.nii.gz', n=10)

            # histogram of permutation testing
            plt.hist(val_list, bins=100)
            plt.xlabel(f'max {fnc_target.__name__} per permutations')
            plt.suptitle(f'{len(val_list)} permutations')
            arba.plot.save_fig(save_folder / 'permute_nr2.pdf')

    return sg_hist, sig_node_list, val_list


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

    sg_hist.reduce_to(1, verbose=True)

    return sg_hist


def permute(fnc_target, feat_sbj, file_tree, n=5000, max_flag=True,
            **kwargs):
    val_list = list()
    for _n in tqdm(range(n), desc='permute'):
        sg_hist = run_single(feat_sbj, file_tree, permute_seed=_n + 1,
                             **kwargs)
        node_val_dict = sg_hist.merge_record.fnc_node_val_list[fnc_target]
        if max_flag:
            val = max(node_val_dict.values())
        else:
            val = min(node_val_dict.values())
        val_list.append(val)

    # ensure that RegionRegress has appropriate feat_sbj ordering
    arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                           sbj_list=file_tree.sbj_list)

    return val_list


def compute_print_dice(mask_estimate, mask_target, save_folder):
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
