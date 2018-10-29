""" constructs PartGraph objects

a PartGraph can operate on one set of images (growing to minimize variance) or
two sets of images (identifyign regions of max kl diff).  this difference in
behavior is encapsulated in the region objects (pnl_segment.adaptive.region)
"""

import nibabel as nib
import numpy as np
from tqdm import tqdm

from pnl_segment.adaptive import region, part_graph
from pnl_segment.point_cloud import point_cloud_ijk


def part_graph_factory(obj, file_tree_dict, history=False):
    """ init PartGraph via img

    Args:
        obj (str): either 'min_var', 'max_kl' or 'max_maha'
        file_tree_dict (dict): keys are grp, values are FileTree
        history (bool): toggles whether part_graph keeps history (see
                        PartGraphHistory)

    Returns:
        part_graph (PartGraph)
    """
    # get appropriate region constructor
    if obj.lower() == 'min_var':
        reg_type = region.RegionMinVar
    elif obj.lower() == 'max_kl':
        reg_type = region.RegionMaxKL
    elif obj.lower() == 'max_maha':
        reg_type = region.RegionMaxMaha
    else:
        raise AttributeError(f'objective not recognized: {obj}')

    # init empty graph
    if history:
        pg = part_graph.PartGraphHistory()
    else:
        pg = part_graph.PartGraph()

    # store
    pg.file_tree_dict = file_tree_dict

    # init obj_fnc
    # todo: wordy, remove this and put into part_graph (maybe even hard code
    # name of fnc: 'get_obj_pair')
    pg.obj_fnc = getattr(reg_type, 'get_obj_pair')

    # check that all file trees have same ref
    ref_list = [ft.ref for ft in file_tree_dict.values()]
    if any(ref_list[0] != ref for ref in ref_list[1:]):
        raise AttributeError('ref space mismatch')

    mask_list = [set(ft.get_mask()) for ft in file_tree_dict.values()]
    ijk_set = set.intersection(*mask_list)
    for ijk in ijk_set:
        # construct pc_ijk
        pc_ijk = point_cloud_ijk.PointCloudIJK(np.atleast_2d(ijk),
                                               ref=ref_list[0])

        # construct region obj
        feat_stat = dict()
        for grp, ft in file_tree_dict.items():
            feat_stat[grp] = ft.ijk_fs_dict[ijk]
        reg = reg_type(pc_ijk, feat_stat=feat_stat)

        # store in graph
        pg.add_node(reg)

    # add edges
    reg_by_ijk = {tuple(r.pc_ijk.x[0, :]): r for r in pg.nodes}
    add_edges(pg, reg_by_ijk)

    return pg


def add_edges(pg, reg_by_ijk, verbose=False, f_constraint=None):
    """ adds edge between each neighboring ijk """

    # build valid_edge, determines if edge between ijk0, ijk1 is valid
    if f_constraint is None:
        def valid_edge(*args, **kwargs):
            # all edges are valid
            return True
    else:
        segment_array = nib.load(str(f_constraint)).get_data()

        def valid_edge(ijk0, ijk1):
            # must belong to same region in segment_array
            return segment_array[tuple(ijk0)] == segment_array[tuple(ijk1)]

    # iterate through offsets of each ijk to find neighbors
    offsets = np.eye(3)
    tqdm_dict = {'desc': 'adding edge',
                 'disable': not verbose}
    for ijk0, reg0 in tqdm(reg_by_ijk.items(), **tqdm_dict):
        for offset in offsets:
            ijk1 = tuple((ijk0 + offset).astype(int))
            if ijk1 in reg_by_ijk.keys() and valid_edge(ijk0, ijk1):
                pg.add_edge(reg0, reg_by_ijk[ijk1])
