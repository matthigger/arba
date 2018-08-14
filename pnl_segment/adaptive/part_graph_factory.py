""" constructs PartGraph objects

a PartGraph can operate on one set of images (growing to minimize variance) or
two sets of images (identifyign regions of max kl diff).  this difference in
behavior is encapsulated in the region objects (pnl_segment.adaptive.region)
"""

import multiprocessing
from itertools import chain

import mh_pytools.paralell
import nibabel as nib
import numpy as np
from pnl_segment.adaptive import region, part_graph, feat_stat
from pnl_segment.point_cloud import point_cloud_ijk
from tqdm import tqdm


def min_var(grp_to_min_var=None, **kwargs):
    """ equivilently, max likelihood under independent / normal assumptions
    """

    def region_init(*args, **kwargs):
        return region.RegionMinVar(*args,
                                   grp_to_min_var=grp_to_min_var,
                                   **kwargs)

    pg = _build_part_graph(**kwargs,
                           region_init=region_init)

    return pg


def _build_part_graph(f_img_dict, region_init, verbose=False,
                      f_mask=None, f_edge_constraint=None, history=False,
                      sbj_thresh=.95):
    """ init PartGraph via img

    Args:
        f_img_dict (dict): keys are grp labels, values are iter, each
                           element of the iter is a list of img from a sbj
                           this is confusing, here's an example:
                           {'tbi': [['FA_sbj1.nii.gz', 'MD_sbj1.nii.gz'], \
                                    ['FA_sbj2.nii.gz', 'MD_sbj2.nii.gz']],
                            'healthy': ... }
                            all should be in same space

        verbose (bool): toggles command line output
        sbj_thresh (float): percentage of sbj needed to include ijk voxel

    Returns:
        part_graph (PartGraph)
    """
    if verbose:
        print('\n' * 2 + 'construct part_graph: load')

    if f_mask is not None:
        raise NotImplementedError('mask not used')
        img_mask = nib.load(str(f_mask))
        mask = img_mask.get_data()

    # get ijk_dict_tree, (ijk_dict_tree[grp][ijk_tuple] = feat_stat)
    ijk_dict_tree = dict()
    for grp, f_img_list_iter in f_img_dict.items():
        if verbose:
            print(f'----{grp}----')
        min_sbj = np.floor(sbj_thresh * len(f_img_list_iter)).astype(int)
        ijk_dict_tree[grp] = get_ijk_dict(f_img_list_iter, verbose=verbose,
                                          min_sbj=min_sbj)

    # get set of all ijk
    ijk_set = set()
    for d in ijk_dict_tree.values():
        ijk_set |= set(d.keys())

    # make sure all affines / shapes are identical
    f_list = [next(chain.from_iterable(f_img_list_iter))
              for f_img_list_iter in f_img_dict.values()]
    affine, _ = check_space(f_list)

    # init empty graph
    if history:
        pg = part_graph.PartGraphHistory()
    else:
        pg = part_graph.PartGraph()

    if verbose:
        print('\n' * 2 + 'construct part_graph: build graph')

    # build graph_nodes
    tqdm_dict = {'desc': 'build nodes (per ijk)',
                 'disable': not verbose,
                 'total': len(ijk_set)}
    reg_by_ijk = dict()
    for ijk in tqdm(ijk_set, **tqdm_dict):
        # construct pc_ijk
        pc_ijk = point_cloud_ijk.PointCloudIJK(np.atleast_2d(ijk),
                                               affine=affine)

        # construct region obj
        try:
            feat_stat = {grp: d[ijk] for grp, d in ijk_dict_tree.items()}
        except KeyError:
            # at least one group doesn't have enough observations @ ijk
            continue
        reg = region_init(pc_ijk, feat_stat=feat_stat)

        # store in graph
        pg.add_node(reg)

        # store by ijk (to add edges later)
        reg_by_ijk[ijk] = reg

    # add edges
    add_edges(pg, reg_by_ijk, f_edge_constraint=f_edge_constraint,
              verbose=verbose)

    if verbose:
        print('finished constructing part_graph' + '\n' * 2)

    return pg


def add_edges(pg, reg_by_ijk, verbose=False, f_edge_constraint=None):
    """ adds edge between each neighboring ijk """
    if f_edge_constraint is not None:
        img_constraint = nib.load(str(f_edge_constraint))
        constraint = img_constraint.get_data()

    # note: we only do "positive" offsets, "negative" ones handled via symmetry
    offsets = np.eye(3)
    tqdm_dict = {'desc': 'adding edge between nodes',
                 'disable': not verbose}
    for ijk1, reg1 in tqdm(reg_by_ijk.items(), **tqdm_dict):
        for offset in offsets:
            ijk2 = tuple((ijk1 + offset).astype(int))

            if ijk2 not in reg_by_ijk.keys():
                # ijk2 never observed, don't add edge
                continue

            reg2 = reg_by_ijk[ijk2]

            if f_edge_constraint is not None:
                if constraint[tuple(ijk1)] != constraint[ijk2]:
                    # ijk1 and ijk2 belong to different "regions", no edge
                    continue

            # add edge between regions
            pg.add_edge(reg1, reg2)


def get_ijk_dict(f_img_list_iter, verbose=False, paralell=False,
                 check_space_flag=True, min_sbj=1):
    """ reads in images, builds feat_stat per ijk location

    Args:
        f_img_list_iter (iter): iter of feature images (str or Path), may be multi
                            dim, for example:
                             [['FA_sbj1.nii.gz', 'MD_sbj1.nii.gz'], \
                              ['FA_sbj2.nii.gz', 'MD_sbj2.nii.gz']],
        verbose (bool): toggles cmd line output
        paralell (bool): toggles paralell data load
        check_space_flag (bool): toggles affine equal check
        min_sbj (int): min num sbj to include ijk

    Returns:
        feat_dict (dict): keys are ijk (tuple), values are FeatStat
    """

    if check_space_flag:
        check_space(chain.from_iterable(f_img_list_iter))

    if paralell:
        raise NotImplementedError('not tested')

        def par_iter():
            x = iter(f_img_list_iter)
            yield next(x), False

        # paralell load
        pool = multiprocessing.Pool()
        res = pool.map_async(get_ijk_dict, par_iter)
        desc = 'load data / compute feat stats (paralell)'
        res_out = mh_pytools.paralell.join(res, verbose=verbose, desc=desc)

        # add data together (from each run)
        tqdm_dict = {'desc': 'join paralell feat stats (per voxel)',
                     'disable': not verbose}
        feat_stat_dict = dict()
        ijk_set = frozenset().union(*[set(d.keys()) for d in res_out])
        for ijk in tqdm(ijk_set, **tqdm_dict):
            feat_stat_dict[ijk] = sum(d[ijk] for d in res_out)
        return feat_stat_dict

    # build feat_dict, keys are ijk positions, values are lists of features
    feat_list = list()
    tqdm_dict = {'total': len(list(f_img_list_iter)),
                 'disable': not verbose,
                 'desc': 'loading feat per sbj'}
    for f_img_list in tqdm(f_img_list_iter, **tqdm_dict):
        # load features
        _feat_list = [nib.load(str(f_img)).get_data() for f_img in f_img_list]
        feat_list.append(np.stack(_feat_list, axis=3))

    # # iterate over valid mask values
    # for ijk in ijk_all:
    #     ijk = tuple(ijk)
    #     x = np.array([x[ijk] for x in feat_list])
    #     feat_dict[ijk].append(x)

    # feat has shape (space, space, space, num_feat, num_sbj)
    feat = np.stack(feat_list, axis=4)

    # compute feat stats
    feat_stat_dict = dict()
    tqdm_dict = {'total': np.prod(feat.shape[:3]),
                 'disable': not verbose,
                 'desc': 'compute feat_stat per voxel'}
    for ijk in tqdm(np.ndindex(feat.shape[:3]), **tqdm_dict):
        # get x, with dim num_feat x num_sbj
        x = feat[ijk[0], ijk[1], ijk[2], :, :]

        # remove col which are not both non-zero
        x = x[:, x.all(axis=0)]

        if x.shape[1] < min_sbj:
            # not enough valid sbj
            continue

        # add it to feat_stat_dict
        obs_greater_dim = x.shape[1] > x.shape[0]
        feat_stat_dict[ijk] = \
            feat_stat.FeatStat.from_array(x, obs_greater_dim)

    return feat_stat_dict


def check_space(f_iter, shape=True, affine=True):
    img_list = [nib.load(str(x)) for x in f_iter]
    affine_list = [x.affine for x in img_list]
    shape_list = [x.shape for x in img_list]

    if affine:
        if any(not np.allclose(a, affine_list[0]) for a in affine_list):
            raise AttributeError('affine mismatch')

    if shape:
        if any(not np.allclose(s, shape_list[0]) for s in shape_list):
            raise AttributeError('shape mismatch')

    return affine_list[0], shape_list[0]
