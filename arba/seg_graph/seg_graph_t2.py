from .seg_graph import SegGraph
from .seg_graph_hist import SegGraphHistory


class SegGraphT2(SegGraphHistory):
    """ keeps track of t-squared distances among all regions through history

    Attributes:
        node_t2_dict (dict): keys are nodes, values are t2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, obj='t2', **kwargs)
        # init node_t2_dict
        self.node_t2_dict = dict()
        for reg in self.nodes:
            ijk = next(iter(reg.pc_ijk))
            self.node_t2_dict[ijk] = reg.t2 * len(reg)

    def merge(self, reg_tuple):
        node_sum = len(self.merge_record)
        reg_sum = super().merge(reg_tuple)

        # record t2 of newest region
        self.node_t2_dict[node_sum] = reg_sum.t2 * len(reg_sum)

        return reg_sum

    def cut_greedy_t2(self):
        """ gets SegGraph of disjoint reg with max t2 which covers volume
        """
        # we 'flip' t2 as _cut_greedy_min() will minimize the given objective
        node_neg_t2_dict = {n: -m for n, m in self.node_t2_dict.items()}

        # get disjoint cover of entire volume which greedily maximizes t2
        node_list = self._cut_greedy_min(node_val_dict=node_neg_t2_dict)

        # build seg graph
        sg = SegGraph(ft_dict=self.ft_dict, _add_nodes=False)
        reg_list = [self.resolve_reg(n) for n in node_list]
        sg.add_nodes_from(reg_list)

        return sg
