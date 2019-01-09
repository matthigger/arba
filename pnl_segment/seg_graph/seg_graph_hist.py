import networkx as nx
import numpy as np

from .seg_graph import SegGraph


class SegGraphHistory(SegGraph):
    """ has a history of which regions were combined

    Attributes:
        tree_history (nx.Digraph): directed seg_graph, from component reg to
                                   sum.  "root" of tree is child, "leafs" are
                                   its farthest ancestors
        reg_history_list (list): stores ordering of calls to combine()
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
            if len(sg.is_sig(alpha=alpha, method='holm')) != len(sg):
                raise RuntimeError('not all regions are significant')

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
