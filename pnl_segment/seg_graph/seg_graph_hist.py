import time

import networkx as nx
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

from .seg_graph import SegGraph
from ..region import RegionMaha
from ..space import PointCloud


class SegGraphHistory(SegGraph):
    """ has a history of how regions were combined from voxel

    Attributes:
        tree_hist (nx.Digraph): directed graph. edges point from child regions
                                (smaller) to parents. leafs are ijk tuples.
                                other nodes are int which describe order they
                                were created (e.g. tree_history[0] is the first
                                node to be created)
        n_combine (int): number of regions which have been combined
        _err_edge_list (SortedList): tuples of (error, (reg0, reg1)) associated
                                     with joining reg0, reg1
        reg_node_dict (dict): keys are regions in self.nodes (from SegGraph,
                              note this is not the full history) values are
                              nodes in tree_hist
        node_pval_dict (dict): keys are nodes, values are pval.  contains full
                               history
    """

    def __init__(self):
        super().__init__()
        self.n_combine = 0
        self.tree_hist = nx.DiGraph()
        self._err_edge_list = SortedList()
        self.reg_node_dict = dict()
        self.node_pval_dict = dict()

    def resolve_space(self, node):
        if isinstance(node, tuple):
            ijk_set = {node}
        else:
            ijk_set = {n for n in nx.ancestors(self.tree_hist, node)
                       if not nx.ancestors(self.tree_hist, n)}
        return PointCloud(ijk_set, ref=self.ref)

    def resolve_reg_iter(self, node):
        pc = self.resolve_space(node)
        for ijk in pc:
            # build dictionary of feature statistics per group
            fs_dict = {grp: self.file_tree_dict[grp].ijk_fs_dict[ijk]
                       for grp in self.file_tree_dict.keys()}

            # build pc
            pc = PointCloud({ijk}, ref=self.ref)

            yield RegionMaha(pc_ijk=pc, fs_dict=fs_dict)

    def resolve_reg(self, node):
        return sum(self.resolve_reg_iter(node))

    def combine(self, reg_tuple):
        """ record combination in tree_hist """
        reg_sum = super().combine(reg_tuple)

        # get new node
        reg_sum_node = self.n_combine
        self.n_combine += 1

        # store new pval
        self.node_pval_dict[reg_sum_node] = reg_sum.pval

        # add new edges in tree_hist
        for reg in reg_tuple:
            reg_node = self.reg_node_dict[reg]
            self.tree_hist.add_edge(reg_node, reg_sum_node)

            # rm reference to reg in reg_node_dict, its no longer in self.nodes
            del self.reg_node_dict[reg]

        # add reference to reg_sum in reg_node_dict
        self.reg_node_dict[reg_sum] = reg_sum_node

        assert len(self.nodes) == len(self.reg_node_dict), 'reg_node_dict err'

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
        pval_node_list = [(p, n) for (n, p) in self.node_pval_dict.items()
                          if p <= alpha]
        pval_node_list = sorted(pval_node_list)
        node_covered = set()

        node_pval_sig_list = list()

        while pval_node_list:
            p, n = pval_node_list.pop(0)
            if n in node_covered:
                continue
            else:
                # add reg to significant regions
                node_pval_sig_list.append((n, p))

                # add all intersecting regions to reg_covered
                node_covered |= {n}
                node_covered |= nx.descendants(self.tree_hist, n)
                node_covered |= nx.ancestors(self.tree_hist, n)

        # get m_max: max num regions for which all are still `significant'
        # under holm-bonferonni.
        # note: it is expected these regions are retested on separate fold
        m_max = np.inf
        for idx, (_, pval) in enumerate(node_pval_sig_list):
            if idx == m_max:
                break
            m_current = np.floor(alpha / pval).astype(int) + idx
            m_max = min(m_current, m_max)

        if m_max < np.inf:
            # resolves nodes
            reg_list = (self.resolve_reg(node)
                        for node, _ in node_pval_sig_list[:m_max])

            # add these regions to output seg_graph
            # note: reg_sig_list is sorted in increasing p value
            sg.add_nodes_from(reg_list)

            # validate all are sig
            assert len(sg.get_sig(alpha=alpha, method='holm')) == len(sg), \
                'not all regions are significant'

        return sg

    def reduce_to(self, num_reg_stop=1, edge_per_step=None, verbose=True,
                  update_period=10, verbose_dbg=False, **kwargs):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (float): (0, 1) how many edges (of those remaining)
                                   to combine in each step.  if not passed 1
                                   edge is combined at all steps.
            verbose (bool): toggles cmd line output
            update_period (float): how often command line updates are given in
                                   verbose_dbg
            verbose_dbg (bool): toggles debug command line output (timing)

        Returns:
            err_list (list): error associated with each step
        """

        if len(self) < num_reg_stop:
            print(f'{len(self)} reg exist, cant reduce to {num_reg_stop}')

        if edge_per_step is not None and not (0 < edge_per_step < 1):
            err_msg = 'edge_per_step not in (0, 1): {edge_per_step}'
            raise AttributeError(err_msg)

        # init edges if need be
        if not self._err_edge_list:
            self._add_err_edge_list(verbose=verbose)

        # init progress stats
        n_neigh_list = list()
        err_list = list()

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
            if not self._err_edge_list or \
                    self._err_edge_list[0][0] > self.err_max:
                print(f'stop @ {len(self)} nodes: wanted {num_reg_stop}')
                break

            # find n edges with min err
            if edge_per_step is not None:
                n = np.ceil(len(self) * edge_per_step).astype(int)
            edge_list, _err_list = self._get_min_n_edges(n)
            err_list += _err_list

            # combine them
            reg_list = list()
            for reg_set in edge_list:
                reg_list.append(self.combine(reg_set))

            # recompute err of new edges to all neighbors of newly combined reg
            edge_list = list()
            for reg in reg_list:
                neighbor_list = list(self.neighbors(reg))
                n_neigh_list.append(len(neighbor_list))
                for reg_neighbor in neighbor_list:
                    edge_list.append((reg, reg_neighbor))
            self._add_err_edge_list(edge_list)

            # command line update
            pbar.update((len_init - len(self)) - pbar.n)

            # output to command line(timing + debug)
            if verbose_dbg and time.time() - last_update > update_period:
                err = np.mean(err_list[-n:])
                print(', '.join([f'n_edge: {len(self._err_edge_list):1.2e}',
                                 f'n_neighbors: {np.mean(n_neigh_list):1.2e}',
                                 f'err: {err:1.2e}']))
                last_update = time.time()
                n_neigh_list = list()

        return err_list

    def _get_min_n_edges(self, n):
        # get edge_per_step edges with minimum error
        edge_list = list()
        err_list = list()
        while len(edge_list) < n:
            if not self._err_edge_list or \
                    self._err_edge_list[0][0] > self.err_max:
                # no more edges
                break

            err, (r1, r2) = self._err_edge_list.pop(0)

            if r1 in self.nodes and r2 in self.nodes:
                err_list.append(err)
                edge_list.append(set((r1, r2)))

        if not edge_list:
            # no edges found at all
            return edge_list, err_list

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

        return edge_list_disjoint, err_list

    def _add_err_edge_list(self, edge_list=None, verbose=False):

        # get list of edges to add (default to
        if edge_list is None:
            self._err_edge_list = SortedList()
            edge_list = self.edges

        # compute error per edge
        tqdm_dict = {'desc': 'compute error per edge',
                     'disable': not verbose}
        for reg_pair in tqdm(edge_list, **tqdm_dict):
            error = reg_pair[0].get_error(*reg_pair)
            if error < self.err_max:
                self._err_edge_list.add((error, reg_pair))
