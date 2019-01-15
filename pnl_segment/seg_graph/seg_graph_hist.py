import time
from collections import namedtuple
from copy import deepcopy

import networkx as nx
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

from .seg_graph import SegGraph
from ..region import Region, RegionMaha
from ..space import PointCloud

# a stand in for RegionMaha in tree_history, lighter memory footprint (lossy).
# see the compress() and extract() methods of SegGraphHistory
RegMahaLight = namedtuple('RegMahaLight', ('size', 'maha', 'pval'))


class SegGraphHistory(SegGraph):
    """ has a history of how regions were combined from voxel

    Attributes:
        tree_history (nx.Digraph): directed graph, each node is a RegMahaLight,
                                   all nodes directed into another had
                                   combine() called on them.  leafs (nodes
                                   with no inward edge) are regions of single
                                   voxels
        reg_history_list (list): list of RegMahaLight, ordered as they were
                                 created
        leaf_ijk_dict (dict): keys are leafs (of tree_history), values are the
                             ijk tuples which describe their location
    """

    @property
    def ref(self):
        return next(iter(self.file_tree_dict.values())).ref

    def __iter__(self):
        """ iterates through all version of seg_graph through its history

        NOTE: every call to next() returns a copy of same object, they share
        memory!

        NOTE: these part_graphs are not connected, neighboring regions do not
        have edges between them (todo: they could be connected)
        """

        # extract_map is a dict.  keys are RegMahaLight, values are their
        # extracted forms.  we initialize it below to include only leafs
        extract_map = {l: self.extract(l) for l in self.leaf_ijk_dict.keys()}

        # build seg_graph, yield it
        sg = SegGraph()
        sg.add_nodes_from(extract_map.values())
        yield sg, None

        for reg_l in self.reg_history_list:
            # get parents of reg_l from tree_history
            reg_parent_l = tuple(self.tree_history.predecessors(reg_l))

            # get extracted versions of parents from extract_map
            reg_parent = tuple(extract_map.pop(_reg_l)
                               for _reg_l in reg_parent_l)

            # combine parents
            reg = sg.combine(reg_parent)

            # update extract_map
            extract_map[reg_l] = reg

            yield sg, reg

    def __init__(self):
        super().__init__()
        self.tree_history = nx.DiGraph()
        self.reg_history_list = list()
        self._obj_edge_list = SortedList()
        self.leaf_ijk_dict = dict()

    def from_file_tree_dict(self, file_tree_dict, copy=True):
        """ returns copy with swapped file_tree_dict

        Args:
            file_tree_dict (dict): keys are groups, values are file_tree
            copy (bool): toggles if output has memory intersection with self

        Returns:
            sg_hist (SegGraphHist):
        """

        # copy as needed
        if copy:
            sg_hist = deepcopy(self)
        else:
            sg_hist = self

        # set new file_tree_dict (used in sg_hist.resolve())
        ref_list = [ft.ref for ft in file_tree_dict.values()]
        if any(ref != self.ref for ref in ref_list):
            raise AttributeError('ref space mismatch')
        sg_hist.file_tree_dict = file_tree_dict

        # reg_map is a dict.  keys are RegMahaLight in old file_tree_dict,
        # values are their corresponding form in new file_tree_dict.  we init
        # on leafs
        reg_map = {leaf: sg_hist.compress(sg_hist.extract(leaf))
                   for leaf, ijk in sg_hist.leaf_ijk_dict.items()}

        # add non leafs to reg_map via sg_hist.__iter__()
        for reg_old, (_, reg_new) in zip(sg_hist.reg_history_list, sg_hist):
            reg_map[reg_old] = reg_new

        if len(reg_map) != len(set(reg_map.values())):
            raise RuntimeError('non unique RegMahaLight via new file_tree')

        # apply reg_map
        nx.relabel_nodes(sg_hist.tree_history, mapping=reg_map)
        sg_hist.reg_history_list = [reg_map[r]
                                    for r in sg_hist.reg_history_list]
        sg_hist._obj_edge_list = SortedList()

        return sg_hist

    def extract(self, reg_light):
        # leaf_set is a set of leafs which are ancestors of reg_light
        ancestors = nx.ancestors(self.tree_history, reg_light) | {reg_light}
        leaf_set = set(self.leaf_ijk_dict.keys()) & ancestors

        if sum(l.size for l in leaf_set) != reg_light.size:
            raise RuntimeError('extract failure: space mismatch')

        # aggregate region per leaf
        reg = 0
        for leaf in leaf_set:
            # build initial space
            ijk = self.leaf_ijk_dict[leaf]
            pc_ijk = PointCloud({ijk}, ref=self.ref)

            # build dictionary of feature statistics per group
            fs_dict = {grp: self.file_tree_dict[grp].ijk_fs_dict[ijk]
                       for grp in self.file_tree_dict.keys()}

            # aggregate
            reg += RegionMaha(pc_ijk=pc_ijk, fs_dict=fs_dict)

        return reg

    def compress(self, reg, make_unique=False):
        # saves bare bone stats for plotting hierarchical tree
        maha = float(reg.maha)
        reg_light = RegMahaLight(size=len(reg), maha=maha, pval=reg.pval)

        if make_unique:
            while reg_light in self.tree_history.nodes:
                # change maha by smallest amount until it is unique ... kludgey
                reg_light = RegMahaLight(size=len(reg),
                                         maha=np.nextafter(maha, 1),
                                         pval=reg.pval)

            # add node
            self.tree_history.add_node(reg_light)

        return reg_light

    def combine(self, reg_tuple):
        """ record combination in tree_history """
        reg_sum = super().combine(reg_tuple)

        # compress (and ensure sum is unique)
        reg_tuple_light = tuple(self.compress(r) for r in reg_tuple)
        reg_sum_light = self.compress(reg_sum, make_unique=True)

        # add edges in tree_history
        for reg_light in reg_tuple_light:
            self.tree_history.add_edge(reg_light, reg_sum_light)

        # record order of calls to combine()
        self.reg_history_list.append(reg_sum_light)

        return reg_sum

    def cut_greedy_sig(self, alpha=.05):
        """ gets seg_graph of disjoint regions with min pval & all reg are sig

        significance is under Bonferonni (equiv Holm if all sig)

        this is achieved by greedily adding min pvalue to output seg graph so
        long as resultant seg_graph contains only significant regions.  after
        each addition, intersecting regions are discarded from search space

        Args:
            alpha (float): significance threshold

        Returns:
            sg (SegGraph): all its regions are disjoint and significant
        """
        # init output seg graph
        sg = SegGraph()
        sg.file_tree_dict = self.file_tree_dict

        # init search space of region to those which have pval <= alpha
        reg_list = [r for r in self.tree_history.nodes if r.pval <= alpha]
        reg_list = sorted(reg_list, key=lambda r: r.pval)
        reg_covered = set()

        reg_sig_list = list()

        while reg_list:
            reg = reg_list.pop(0)
            if reg in reg_covered:
                continue
            else:
                # add reg to significant regions
                reg_sig_list.append(reg)

                # add all intersecting regions to reg_covered
                reg_covered |= {reg}
                reg_covered |= nx.descendants(self.tree_history, reg)
                reg_covered |= nx.ancestors(self.tree_history, reg)

        # get m_max: max num regions for which all are still `significant'
        # under holm-bonferonni.
        # note: it is expected these regions are retested on separate fold
        m_max = np.inf
        for idx, reg in enumerate(reg_sig_list):
            if idx == m_max:
                break
            m_current = np.floor(alpha / reg.pval).astype(int) + idx
            m_max = min(m_current, m_max)

        if m_max < np.inf:
            # resolves nodes
            reg_list = (self.extract(r) for r in reg_sig_list[:m_max])

            # add these regions to output seg_graph
            # note: reg_sig_list is sorted in increasing p value
            sg.add_nodes_from(reg_list)

            # validate all are sig
            if len(sg.get_sig(alpha=alpha, method='holm')) != len(sg):
                raise RuntimeError('not all regions are significant')

        return sg

    def harmonize_via_add(self, *args, **kwargs):
        """ ensure harmonization done before combine() has been called
        """
        x = super().harmonize_via_add(apply=False)

        if self.reg_history_list:
            raise RuntimeError('harmonize_via_add() called after combine()')

        return x

    def reduce_to(self, num_reg_stop=1, edge_per_step=None, verbose=True,
                  update_period=10, verbose_dbg=False, **kwargs):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (float): (0, 1) how many edges (of those remaining)
                                   to combine in each step.  if not passed 1
                                   edge is combined at all steps.
            verbose (bool): toggles cmd line output
            update_period (float): how often command line updates are given

        Returns:
            obj_list (list): objective fnc at each combine
        """

        if len(self) < num_reg_stop:
            print(f'{len(self)} reg exist, cant reduce to {num_reg_stop}')

        if edge_per_step is not None and not (0 < edge_per_step < 1):
            err_msg = 'edge_per_step not in (0, 1): {edge_per_step}'
            raise AttributeError(err_msg)

        # init edges if need be
        if not self._obj_edge_list:
            self._add_obj_edge_list(verbose=verbose)

        # init progress stats
        n_neigh_list = list()
        obj_list = list()

        # init
        pbar = tqdm(total=len(self) - num_reg_stop,
                    desc='combining edges',
                    disable=not verbose)

        # combine edges until only n regions left
        len_init = len(self)
        last_update = time.time()
        n = 1
        while len(self) > num_reg_stop:
            # break early if no more valid edges available
            if not self._obj_edge_list or \
                    self._obj_edge_list[0][0] > self.obj_fnc_max:
                print(f'stop @ {len(self)} nodes: wanted {num_reg_stop}')
                break

            # find n edges with min obj
            if edge_per_step is not None:
                n = np.ceil(len(self) * edge_per_step).astype(int)
            edge_list, _obj_list = self._get_min_n_edges(n)
            obj_list += _obj_list

            # combine them
            reg_list = list()
            for reg_set in edge_list:
                reg_list.append(self.combine(reg_set))

            # recompute obj of new edges to all neighbors of newly combined reg
            edge_list = list()
            for reg in reg_list:
                neighbor_list = list(self.neighbors(reg))
                n_neigh_list.append(len(neighbor_list))
                for reg_neighbor in neighbor_list:
                    edge_list.append((reg, reg_neighbor))
            self._add_obj_edge_list(edge_list)

            # command line update
            pbar.update((len_init - len(self)) - pbar.n)

            # output to command line(timing + debug)
            if verbose_dbg and time.time() - last_update > update_period:
                obj = np.mean(obj_list[-n:])
                print(', '.join([f'n_edge: {len(self._obj_edge_list):1.2e}',
                                 f'n_neighbors: {np.mean(n_neigh_list):1.2e}',
                                 f'obj: {obj:1.2e}']))
                last_update = time.time()
                n_neigh_list = list()

        return obj_list

    def _get_min_n_edges(self, n):
        # get edge_per_step edges with minimum objective
        edge_list = list()
        obj_list = list()
        while len(edge_list) < n:
            if not self._obj_edge_list or \
                    self._obj_edge_list[0][0] > self.obj_fnc_max:
                # no more edges
                break

            obj, (r1, r2) = self._obj_edge_list.pop(0)

            if r1 in self.nodes and r2 in self.nodes:
                obj_list.append(obj)
                edge_list.append(set((r1, r2)))

        if not edge_list:
            # no edges found at all
            return edge_list, obj_list

        # some edges may intersect each other, join these region sets
        edge_list_disjoint = []
        while edge_list:
            # get first reg_set
            reg_set = edge_list.pop(0)

            # find first intersection
            reg_set_int = next((r_set for r_set in edge_list if
                                r_set.intersection(reg_set)), None)

            if reg_set_int:
                # if it exists, add reg_set into the intersecting set
                reg_set_int |= reg_set
            else:
                # disjoint, add to disjoint list
                edge_list_disjoint.append(reg_set)

        return edge_list_disjoint, obj_list

    def _add_obj_edge_list(self, edge_list=None, verbose=False):

        # get list of edges to add (default to
        if edge_list is None:
            self._obj_edge_list = SortedList()
            edge_list = self.edges

        # compute objective per edge
        tqdm_dict = {'desc': 'compute obj per edge',
                     'disable': not verbose}
        for reg_pair in tqdm(edge_list, **tqdm_dict):
            obj = Region.get_error_delta(*reg_pair)
            if obj < self.obj_fnc_max:
                self._obj_edge_list.add((obj, reg_pair))
