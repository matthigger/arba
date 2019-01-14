import networkx as nx

from .seg_graph_hist import SegGraphHistory


class SegGraphHistoryLight(SegGraphHistory):
    """ the history is 'light' in terms of memory, helpful for full brain

    observe that cut_greedy_sig() nominates regions to represent a set of
    regions which have higher pval.  in this, we prune away any 'middle' nodes
    whose inclusion does not effect the operation of cut_greedy_sig(). A
    middle node is not a leaf and has a descendent (larger region) with a
    smaller pval.

    note: leafs are needed to keep space definitions in leaf_ijk_dict

    Attributes:
        reg_minpval_dict (dict): keys are non-leaf nodes in tree_history, vals
                                 are the minimum pval among all its ancestors
                                 as well as itself

    todo: I think theres a way to make this even lighter, can we include
    predecessors in the superfluous 'middle' node set too?  The same conditions
    arent sufficient though, it'll interrupt how cut_greedy_sig() works ...
    """

    def __init__(self):
        super().__init__()
        self.reg_minpval_dict = dict()
        self._clean_per_combine = True

    def get_minpval(self, reg):
        try:
            # dict stores min pval of all ancestors (and self)
            return self.reg_minpval_dict[reg]
        except KeyError:
            # if not seen, reg is a leaf in tree, minpval is its own pval
            return reg.pval

    def reduce_to(self, *args, **kwargs):
        """ cleans reg_history_list @ end once """

        self._clean_per_combine = False
        x = super().reduce_to(*args, **kwargs)

        self._clean_per_combine = True
        self.clean_reg_history_list()

        return x

    def combine(self, reg_iter):
        """ record combination in tree_history """
        reg_sum = super().combine(reg_iter)

        # store minimum pval
        pval_ancestor = min(self.get_minpval(r) for r in reg_iter)
        pval_min = min(pval_ancestor, reg_sum.pval)
        self.reg_minpval_dict[reg_sum] = pval_min

        if pval_min < pval_ancestor:
            # all ancestors are leaf or middle nodes
            reg_middle = set()
            reg_leaf = set()
            for reg in nx.ancestors(self.tree_history, reg_sum):
                if nx.ancestors(self.tree_history, reg):
                    # if it has ancestors, its a middle node
                    reg_middle.add(reg)
                else:
                    # if no ancestors, its a leaf
                    reg_leaf.add(reg)

            # remove middle nodes
            for reg in reg_middle:
                assert reg not in self.nodes, 'removing non-historical node'
                self.tree_history.remove_node(reg)
                del self.reg_minpval_dict[reg]

            # rewire all leafs to their new representative: reg_sum
            for reg in reg_leaf:
                self.tree_history.add_edge(reg, reg_sum)

        if self._clean_per_combine:
            self.clean_reg_history_list()

        return reg_sum

    def clean_reg_history_list(self):
        self.reg_history_list = [r for r in self.reg_history_list
                                 if r in self.tree_history]
