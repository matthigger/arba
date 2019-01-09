import multiprocessing
import time

import networkx as nx
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

import mh_pytools.parallel
from .seg_graph import SegGraph
from ..region import Region


class SegGraphHistory(SegGraph):
    """ has a history of which regions were combined

    Attributes:
        tree_history (nx.Digraph): directed seg_graph, from component reg to
                                   sum.  "root" of tree is child, "leafs" are
                                   its farthest ancestors
        reg_history_list (list): stores ordering of calls to combine()
        leaf_pc_dict (dict): keys are leafs (of tree_history), values are the
                             point_clouds which describe their location. by
                             storing location for only the leafs we avoid doing
                             so for all their descendents, see space_drop() and
                             space_resolve() methods.
    """

    def __iter__(self):
        """ iterates through all version of seg_graph through its history

        NOTE: these part_graphs are not connected, neighboring regions do not
        have edges between them.
        """

        def build_part_graph(reg_set):
            # build seg_graph
            seg_graph = SegGraph()
            seg_graph.obj_fnc_max = np.inf
            seg_graph.add_nodes_from(reg_set)
            return seg_graph

        # get all leaf nodes (reg without ancestors in tree_history)
        reg_set = set(self.leaf_iter)

        yield build_part_graph(reg_set)

        for reg_next in self.reg_history_list:
            # subtract the kids, add the next node
            reg_kids = set(self.tree_history.predecessors(reg_next))
            reg_set = (reg_set - reg_kids) | {reg_next}

            yield build_part_graph(reg_set)

    @property
    def leaf_iter(self):
        return (r for r in self.tree_history.nodes
                if not nx.ancestors(self.tree_history, r))

    @property
    def root_iter(self):
        return (r for r in self.tree_history
                if not nx.descendants(self.tree_history, r))

    def __init__(self):
        super().__init__()
        self.tree_history = nx.DiGraph()
        self.reg_history_list = list()
        self.leaf_pc_dict = dict()

    def from_file_tree_dict(self, file_tree_dict):
        sg_hist = super().from_file_tree_dict(file_tree_dict)

        # build map of old regions to new (those from new file_tree_dict)
        reg_map = {reg: reg.from_file_tree_dict(file_tree_dict)
                   for reg in self.tree_history.nodes}

        # add edges which mirror original
        new_edges = ((reg_map[r0], reg_map[r1])
                     for r0, r1 in self.tree_history.edges)
        sg_hist.tree_history.add_edges_from(new_edges)

        # map reg_history_list
        sg_hist.reg_history_list = [reg_map[r] for r in self.reg_history_list]

        return sg_hist

    def combine(self, reg_iter):
        """ record combination in tree_history """
        reg_sum = super().combine(reg_iter)

        for reg in reg_iter:
            self.tree_history.add_edge(reg, reg_sum)

        self.reg_history_list.append(reg_sum)
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
        # note: it is expected these regions are retested on seperate fold
        m_max = np.inf
        for idx, reg in enumerate(reg_sig_list):
            if idx == m_max:
                break
            m_current = np.floor(alpha / reg.pval).astype(int) + idx
            m_max = min(m_current, m_max)

        if m_max < np.inf:
            # add these regions to output seg_graph
            # note: reg_sig_list is sorted in increasing p value
            sg.add_nodes_from(reg_sig_list[:m_max])

            # validate all are sig
            if len(sg.get_sig(alpha=alpha, method='holm')) != len(sg):
                raise RuntimeError('not all regions are significant')

        # resolve space of nodes
        self.space_resolve(sg.nodes)

        return sg

    def harmonize_via_add(self, apply=True):
        """ adds, uniformly, to each grp to ensure same average over whole reg

        note: the means meet at the weighted average of their means (more
        observations => smaller movement)

        Returns:
            mu_offset_dict (dict): keys are grp, values are offsets of average
        """
        mu_offset_dict = super().harmonize_via_add(apply=False)

        # add to all regions
        if apply:
            node_set = set(self.tree_history.nodes) | set(self.nodes)
            for r in node_set:
                for grp, mu in mu_offset_dict.items():
                    r.fs_dict[grp].mu += mu
                r.reset()

        return mu_offset_dict

    def reduce_to(self, num_reg_stop=1, edge_per_step=None, verbose=True,
                  par_thresh=False, update_period=10, **kwargs):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (float): (0, 1) how many edges (of those remaining)
                                   to combine in each step.  if not passed 1
                                   edge is combined at all steps.
            verbose (bool): toggles cmd line output
            par_thresh (int): min threshold for paralell computation of edge
                              weights
            update_period (float): how often command line updates are given

        Returns:
            obj_list (list): objective fnc at each combine
        """

        if len(self) < num_reg_stop:
            print(f'{len(self)} reg exist, cant reduce to {num_reg_stop}')

        if edge_per_step is not None and not (0 < edge_per_step < 1):
            err_msg = 'edge_per_step not in (0, 1): {edge_per_step}'
            raise AttributeError(err_msg)

        # drop space in regions (see space_drop())
        self.space_drop()

        # init edges if need be
        if self._obj_edge_list is None:
            self._add_obj_edge_list(verbose=verbose, max_size_rat=np.inf)

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
                print(f'stop: no valid edges {len(self)} (obj:{num_reg_stop})')
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
            self._add_obj_edge_list(edge_list, par_flag=par_thresh)

            # command line update
            pbar.update((len_init - len(self)) - pbar.n)

            # output to command line(timing + debug)
            if verbose and time.time() - last_update > update_period:
                obj = np.mean(obj_list[-n:])
                print(', '.join([f'n_edge: {len(self._obj_edge_list):1.2e}',
                                 f'n_neighbors: {np.mean(n_neigh_list):1.2e}',
                                 f'obj: {obj:1.2e}']))
                last_update = time.time()
                n_neigh_list = list()

        # add space of regions
        self.space_resolve()

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

    def _add_obj_edge_list(self, edge_list=None, par_flag=False,
                           verbose=False):

        if edge_list is None:
            self._obj_edge_list = SortedList()
            edge_list = self.edges

        if not isinstance(par_flag, bool):
            # a threshhold, not bool, was passed
            par_flag = len(edge_list) > par_flag

        if par_flag:
            # compute (parallel) objective per edge
            raise NotImplementedError
            pool = multiprocessing.Pool()
            res = pool.starmap_async(self.obj_fnc, edge_list)
            obj_list = mh_pytools.parallel.join(pool, res,
                                                desc='compute obj per edge (par)',
                                                verbose=verbose)

            # add to obj_edge_list
            for obj, reg_pair in zip(obj_list, edge_list):
                if obj < self.obj_fnc_max:
                    self._obj_edge_list.add((obj, reg_pair))

        else:
            # compute objective per edge
            tqdm_dict = {'desc': 'compute obj per edge',
                         'disable': not verbose}
            for reg_pair in tqdm(edge_list, **tqdm_dict):
                obj = Region.get_error_delta(*reg_pair)
                if obj < self.obj_fnc_max:
                    self._obj_edge_list.add((obj, reg_pair))

    def space_drop(self):
        if not self.leaf_pc_dict:
            # get set of all leafs (including leafs to be)
            leaf_set = set(self.leaf_iter)
            leaf_set |= (set(self.nodes) - set(self.tree_history.nodes))

            # record space of all leafs
            self.leaf_pc_dict = {reg: reg.pc_ijk for reg in leaf_set}

        # remove space of each region
        for reg in self.nodes:
            reg.pc_ijk = set()

    def space_resolve(self, reg_list=None):
        if reg_list is None:
            reg_list = list(self.nodes)

        for reg in reg_list:
            # reg_constit_set is a set of regions contained in reg
            reg_constit_set = {reg}
            if reg in self.tree_history.nodes:
                reg_constit_set |= set(nx.ancestors(self.tree_history, reg))

            # collect point_clouds of all constituent regions
            pc_list = list()
            for r in reg_constit_set:
                try:
                    pc_list.append(self.leaf_pc_dict[r])
                except KeyError:
                    # not a leaf, its point cloud is redundant
                    continue

            if not pc_list:
                raise RuntimeError('space_resolve failure')

            # build point_cloud as union of all leafs.  we can't just use
            # set.union as reg.pc_ijk is of type PointCloud (which preserves
            # reference space)
            reg.pc_ijk = pc_list[0]
            if len(pc_list) > 1:
                reg.pc_ijk |= set.union(*pc_list[1:])
