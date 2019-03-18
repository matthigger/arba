import numpy as np

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
            node = self.merge_record.ijk_leaf_dict[ijk]
            self.node_t2_dict[node] = reg.t2 * len(reg)

    def merge(self, reg_tuple):
        node_sum = len(self.merge_record.nodes)
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
        sg = SegGraph(file_tree=self.file_tree, grp_sbj_dict=self.grp_sbj_dict,
                      _add_nodes=False)
        reg_list = list()
        for n in node_list:
            reg = self.merge_record.resolve_node(n,
                                                 file_tree=self.file_tree,
                                                 grp_sbj_dict=self.grp_sbj_dict)
            reg_list.append(reg)
        sg.add_nodes_from(reg_list)

        return sg

    def get_max_t2_array(self):
        """ returns volume of max t2 per voxel

        Returns:
            max_t2 (array): maximum t2 observed per each voxel
        """
        node_list = self._cut_greedy_min(node_val_dict=self.node_t2_dict)

        max_t2 = np.zeros(self.file_tree.ref.shape)
        for n in node_list:
            mask = self.merge_record.get_pc(n).to_mask()
            max_t2[mask] = self.node_t2_dict[n]
            
        return max_t2
