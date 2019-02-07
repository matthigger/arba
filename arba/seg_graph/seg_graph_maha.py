from .seg_graph import SegGraph
from .seg_graph_hist import SegGraphHistory


class SegGraphMaha(SegGraphHistory):
    """ keeps track of mahalanobis among all regions through history

    Attributes:
        node_maha_dict (dict): keys are nodes, values are maha
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, obj='maha', **kwargs)
        # init node_maha_dict
        self.node_maha_dict = dict()
        for reg in self.nodes:
            ijk = next(iter(reg.pc_ijk))
            self.node_maha_dict[ijk] = reg.maha * len(reg)

    def combine(self, reg_tuple):
        reg_sum = super().combine(reg_tuple)

        # record maha of newest region
        node_sum = self.reg_node_dict[reg_sum]
        self.node_maha_dict[node_sum] = reg_sum.maha * len(reg_sum)

        return reg_sum

    def cut_greedy_maha(self):
        """ gets SegGraph of disjoint reg with max maha which covers volume
        """
        # we 'flip' maha as _cut_greedy_min() will minimize the given objective
        node_neg_maha_dict = {n: -m for n, m in self.node_maha_dict.items()}

        # get disjoint cover of entire volume which greedily maximizes maha
        node_list = self._cut_greedy_min(node_val_dict=node_neg_maha_dict)

        # build seg graph
        sg = SegGraph(obj=self.reg_type, file_tree_dict=self.file_tree_dict,
                      _add_nodes=False)
        reg_list = [self.resolve_reg(n) for n in node_list]
        sg.add_nodes_from(reg_list)

        return sg
