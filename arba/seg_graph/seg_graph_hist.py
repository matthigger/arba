import time

import networkx as nx
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

from .merge_record import MergeRecord
from .seg_graph import SegGraph
from ..region import RegionT2Ward


class SegGraphHistory(SegGraph):
    """ manages the merging of regions to optimize some obj_fnc

    todo: this should have a seg_graph and a merge_record, not be both (FQ)

    Attributes:
        merge_record (MergeRecord)
        _err_edge_list (SortedList): tuples of (error, (reg0, reg1)) associated
                                     with joining reg0, reg1

        todo: rm the two attributes below, move history into merge_record
        reg_node_dict (dict): keys are regions in self.nodes (from SegGraph,
                              note this is not the full history) values are
                              nodes in merge_record
        node_pval_dict (dict): keys are nodes, values are pval.  contains full
                               history
    """

    def __init__(self, *args, obj_fnc=None, **kwargs):
        super().__init__(*args, **kwargs)

        ft = next(iter(self.ft_dict.values()))
        self.merge_record = MergeRecord(ft.mask, ref=ft.ref)

        self.obj_fnc = obj_fnc
        self._err_edge_list = None
        if self.obj_fnc is None:
            self.obj_fnc = RegionT2Ward.get_error_tr

        # init reg_node_dict, node_pval_dict
        self.reg_node_dict = dict()
        self.node_pval_dict = dict()
        for reg in self.nodes:
            assert len(reg) == 1, 'invalid region, must be a single voxel'
            ijk = next(iter(reg.pc_ijk))
            self.reg_node_dict[reg] = ijk
            node = self.merge_record.ijk_leaf_dict[ijk]
            self.node_pval_dict[node] = reg.pval

    def from_ft_dict(self, ft_dict):
        sg_hist = super().from_ft_dict(ft_dict)
        sg_hist.merge_record = self.merge_record

        return sg_hist

    def merge(self, reg_tuple):
        """ record combination in merge_record """
        node_sum = len(self.merge_record)
        self.merge_record.merge(reg_tuple=reg_tuple)

        reg_sum = super().merge(reg_tuple)

        self.reg_node_dict[reg_sum] = node_sum
        self.node_pval_dict[node_sum] = reg_sum.pval

        return reg_sum

    def _cut_greedy_min(self, node_val_dict):
        """ gets SegGraph of disjoint reg which minimize val

        NOTE: the resultant node_list covers node_val_dict, ie each node in
        node_val_dict has some ancestor, descendant or itself in node_list

        Args:
            node_val_dict (dict): keys are nodes, values are associated values
                                  to be minimized

        Returns:
             node_list (list): nodes have minimum val, are disjoint
        """
        # sort all nodes
        node_list_sorted = sorted(node_val_dict.keys(), key=node_val_dict.get)

        # init
        node_covered = set()
        node_list = list()

        while node_list_sorted:
            n = node_list_sorted.pop(0)
            if n in node_covered:
                continue
            else:
                # add reg to significant regions
                node_list.append(n)

                # add all intersecting regions to reg_covered (no need to add
                # n, its only in node_list_sorted once)
                node_covered |= nx.descendants(self.merge_record, n)
                node_covered |= nx.ancestors(self.merge_record, n)

        return node_list

    def cut_greedy_sig(self, alpha=.05):
        """ gets SegGraph of disjoint reg with min pval such that all are sig

        significance is under Bonferonni (equiv Holm if all sig)

        this is achieved by greedily adding region of min pval to seg_graph so
        long as resultant seg_graph contains only significant regions.  after
        each addition, intersecting regions are discarded from search space

        Args:
            alpha (float): significance threshold

        Returns:
            sg (SegGraph): all its regions are disjoint and significant
        """

        # init output seg graph
        sg = SegGraph(ft_dict=self.ft_dict, _add_nodes=False)

        # get spanning, disjoint and minimal pval node_list
        node_pval_dict = {n: p for n, p in self.node_pval_dict.items()
                          if p <= alpha}
        node_list = self._cut_greedy_min(node_val_dict=node_pval_dict)
        node_pval_list_sig = [(n, node_pval_dict[n]) for n in node_list]

        # get m_max: max num regions for which all are still `significant'
        # under holm-bonferonni.
        # note: it is expected these regions are retested on separate fold
        m_max = np.inf
        for idx, (_, pval) in enumerate(node_pval_list_sig):
            if idx == m_max:
                break
            m_current = np.floor(alpha / pval).astype(int) + idx
            m_max = min(m_current, m_max)

        if m_max < np.inf:
            # get regions associated with each node
            node_reg_dict = {n: r for r, n in self.reg_node_dict.items()}
            reg_list = [node_reg_dict[n]
                        for n, _ in node_pval_list_sig[:m_max]]

            # add these regions to output seg_graph
            # note: reg_sig_list is sorted in increasing p value
            sg.add_nodes_from(reg_list)

            # validate all are sig
            assert len(sg.get_sig(alpha=alpha, method='holm')) == len(sg), \
                'not all regions are significant'

        return sg

    def get_sig(self, *args, **kwargs):
        # compose self.reg_node_dict and self.node_pval_dict
        _reg_pval_dict = {reg: self.node_pval_dict[n]
                          for reg, n in self.reg_node_dict.items()}
        return super().get_sig(*args, _reg_pval_dict=_reg_pval_dict, **kwargs)

    def reduce_to(self, num_reg_stop=1, edge_per_step=None, verbose=True,
                  update_period=10, verbose_dbg=False, **kwargs):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (float): (0, 1) how many edges (of those remaining)
                                   to merge in each step.  if not passed 1
                                   edge is combined at all steps.
            verbose (bool): toggles cmd line output
            update_period (float): how often command line updates are given in
                                   verbose_dbg
            verbose_dbg (bool): toggles debug command line output (timing)

        Returns:
            err_list (list): error associated with each step
        """
        # todo: clean this method

        self._err_edge_list = SortedList()

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

        # merge edges until only n regions left
        len_init = len(self)
        last_update = time.time()
        n = 1
        while len(self) > num_reg_stop:
            # break early if no more valid edges available
            if not self._err_edge_list:
                if verbose_dbg:
                    print(f'stop @ {len(self)} nodes: wanted {num_reg_stop}')
                break

            # find n edges with min err
            if edge_per_step is not None:
                n = np.ceil(len(self) * edge_per_step).astype(int)
            edge_list, _err_list = self._get_min_n_edges(n)
            err_list += _err_list

            # merge them
            reg_list = list()
            for reg_set in edge_list:
                reg_list.append(self.merge(reg_set))

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

        # cleanup
        self._err_edge_list = None

        return err_list

    def _get_min_n_edges(self, n):
        # get edge_per_step edges with minimum error
        edge_list = list()
        err_list = list()
        while len(edge_list) < n:
            if not self._err_edge_list:
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
            error = self.obj_fnc(*reg_pair)
            self._err_edge_list.add((error, reg_pair))
