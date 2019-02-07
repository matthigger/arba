from copy import deepcopy

import networkx as nx
import numpy as np
import time
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

    def __iter__(self):
        """ returns SegGraph throughout the reduce_to() process

        NOTE: each yield returns the same object, this method calls combine()
        as was done in tree_hist ...

        Yields:
            sg (SegGraph): SegGraph
            node (int): node which was just added to SegGraph (or None on 1st)
            reg_sum: region which was just added to SegGraph (or None on 1st)
        """
        # init SegGraph
        leaf_set = set(n for n in self.tree_hist
                       if not nx.ancestors(self.tree_hist, n))
        sg = SegGraph(obj=self.reg_type, file_tree_dict=self.file_tree_dict,
                      ijk_set=leaf_set)

        yield sg, None, None

        # node_list are sorted by which was created first, doesnt include leafs
        node_list = sorted(set(self.tree_hist.nodes) - leaf_set)

        # node_reg_dict contains a
        node_reg_dict = {next(iter(reg.pc_ijk)): reg for reg in sg.nodes}
        for node in node_list:
            # lookup which regions to combine
            node_tuple = tuple(self.tree_hist.predecessors(node))
            reg_tuple = [node_reg_dict[node] for node in node_tuple]

            # combine
            reg_sum = sg.combine(reg_tuple)

            # update node_reg_dict
            for n in node_tuple:
                del node_reg_dict[n]
            node_reg_dict[node] = reg_sum

            yield sg, node, reg_sum

    def __reduce_ex__(self, *args, **kwargs):
        self._err_edge_list = SortedList()
        return super().__reduce_ex__(*args, **kwargs)

    def __init__(self, *args, err_max=np.inf, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_combine = 0
        self.tree_hist = nx.DiGraph()
        self._err_edge_list = SortedList()
        self.err_max = err_max

        # init reg_node_dict, node_pval_dict
        self.reg_node_dict = dict()
        self.node_pval_dict = dict()
        for reg in self.nodes:
            assert len(reg) == 1, 'invalid region, must be a single voxel'
            ijk = next(iter(reg.pc_ijk))
            self.reg_node_dict[reg] = ijk
            self.node_pval_dict[ijk] = reg.pval

        # init tree_hist with leafs. other nodes added in combine(), no promise
        # that all leafs called in combine()
        self.tree_hist.add_nodes_from(self.node_pval_dict.keys())

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

    def from_file_tree_dict(self, file_tree_dict):
        sg_hist = super().from_file_tree_dict(file_tree_dict)

        # copy fields from self
        sg_hist.n_combine = self.n_combine
        sg_hist.tree_hist = deepcopy(self.tree_hist)
        sg_hist.err_max = self.err_max

        # update reg_node_dict and node_pval_dict to new file_tree_dict
        node_reg_dict, _ = sg_hist.resolve_hist()
        sg_hist.reg_node_dict = {node_reg_dict[n]: n
                                 for n in self.reg_node_dict.values()}
        sg_hist.node_pval_dict = {node: reg.pval
                                  for node, reg in node_reg_dict.items()}

        return sg_hist

    def resolve_hist(self):
        """ returns a copy of tree_hist where each node is replaced by region

        NOTE: for large tree_hist, this will use a lot of memory

        Returns:
            node_reg_dict (dict): keys are nodes, values are regions
            tree_hist_resolve (nx.DiGraph): each node replaced with resolved
                                            version
        """
        # initialize iterator over all historical SegGraph
        pg_res_iter = iter(self)

        # initialize node_reg_dict from the leafs
        pg, _, _ = next(pg_res_iter)
        node_reg_dict = {next(iter(reg.pc_ijk)): reg for reg in pg.nodes}

        # build non leaf entries of node_reg_dict
        for _, node, reg in pg_res_iter:
            node_reg_dict[node] = reg

        # map nodes
        tree_hist_resolve = deepcopy(self.tree_hist)
        tree_hist_resolve = nx.relabel_nodes(tree_hist_resolve,
                                             mapping=node_reg_dict)

        return node_reg_dict, tree_hist_resolve

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

    def match(self, ijk_set):
        """ given a set of ijk, returns the corresponding node (if it exists)

        we choose an arbitrary ijk in ijk_set (a leaf in tree_hist).  we follow
        edges in the tree until cover equals a leaf not in ijk_set
        """

        raise NotImplementedError('not tested')

        # init a leaf_set
        leaf_set = set(n for n in self.tree_hist.nodes
                       if not nx.ancestors(self.tree_hist, n))

        # init node and ijk_cover
        node = next(iter(ijk_set))
        ijk_cover = set(nx.ancestors(self.tree_hist, node)) | {node}
        ijk_cover &= leaf_set

        while True:
            if ijk_cover - ijk_set:
                # node doesnt exist
                raise RuntimeError('no matching node found')

            elif not ijk_set - ijk_cover:
                # node found (node_cover == ijk_set)
                return node

            # go along outward edge
            node_next = list(self.tree_hist.successors(node))
            assert len(node_next) == 1, 'non tree tree_hist ...'
            node = node_next[0]
            ijk_cover = set(nx.ancestors(self.tree_hist, node)) & leaf_set

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
                node_covered |= nx.descendants(self.tree_hist, n)
                node_covered |= nx.ancestors(self.tree_hist, n)

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
        sg = SegGraph(obj=self.reg_type, file_tree_dict=self.file_tree_dict,
                      _add_nodes=False)

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
            # resolves nodes
            reg_list = [self.resolve_reg(node)
                        for node, _ in node_pval_list_sig[:m_max]]

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
                if verbose_dbg:
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
