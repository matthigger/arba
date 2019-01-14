import nibabel as nib
import numpy as np
from tqdm import tqdm

from .seg_graph_hist import SegGraphHistory
from .seg_graph_hist_light import SegGraphHistoryLight
from ..region import RegionMinVar, RegionKL, RegionMaha
from ..space import PointCloud


def seg_graph_factory(obj, file_tree_dict, light_memory=False, **kwargs):
    """ init PartGraph via img

    Args:
        obj (str): either 'min_var', 'kl' or 'maha'
        file_tree_dict (dict): keys are grp, values are FileTree
        light_memory (bool): toggles `light' memory, see SegGraphHistoryLight

    Returns:
        seg_graph (PartGraph)
    """
    # get appropriate region constructor
    obj_dict = {'min_var': RegionMinVar,
                'kl': RegionKL,
                'maha': RegionMaha}
    reg_type = obj_dict[obj.lower()]

    # init empty seg_graph
    if light_memory:
        sg = SegGraphHistoryLight()
    else:
        sg = SegGraphHistory()

    # store
    sg.file_tree_dict = file_tree_dict

    # check that all file trees have same ref
    ref_list = [ft.ref for ft in file_tree_dict.values()]
    if any(ref_list[0] != ref for ref in ref_list[1:]):
        raise AttributeError('ref space mismatch')

    # ensure file_tree is loaded
    for ft in file_tree_dict.values():
        if not ft.ijk_fs_dict.keys():
            ft.load()

    ijk_set = (set(ft.ijk_fs_dict.keys()) for ft in file_tree_dict.values())
    ijk_set = set.intersection(*ijk_set)
    for ijk in ijk_set:
        # construct pc_ijk
        pc_ijk = PointCloud({tuple(ijk)}, ref=ref_list[0])

        # construct region obj
        fs_dict = dict()
        for grp, ft in file_tree_dict.items():
            fs_dict[grp] = ft.ijk_fs_dict[ijk]
        reg = reg_type(pc_ijk, fs_dict=fs_dict)

        # store in seg_graph
        sg.add_node(reg)

    # add edges
    reg_by_ijk = {next(iter(r.pc_ijk)): r for r in sg.nodes}
    add_edges(sg, reg_by_ijk)

    return sg


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
