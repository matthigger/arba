import numpy as np

from .seg_graph import SegGraph
from .seg_graph_hist import SegGraphHistory


class SegGraphHistT2(SegGraphHistory):
    """ keeps track of t2 among all regions through history

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
            self.node_t2_dict[node] = reg.t2

    def merge(self, reg_tuple):
        node_sum = len(self.merge_record.nodes)
        reg_sum = super().merge(reg_tuple)

        # record pval of newest region
        self.node_t2_dict[node_sum] = reg_sum.t2

        return reg_sum

    def cut_greedy_t2(self):
        """ gets SegGraph of disjoint reg with max t2 which covers volume
        """
        # get disjoint cover of entire volume which greedily minimizes pval
        node_list = self._cut_greedy(node_val_dict=self.node_t2_dict,
                                     max_flag=True)

        # build seg graph
        sg = SegGraph(file_tree=self.file_tree, split=self.split,
                      _add_nodes=False)
        reg_list = list()
        for n in node_list:
            reg = self.merge_record.resolve_node(n,
                                                 file_tree=self.file_tree,
                                                 split=self.split)
            reg_list.append(reg)
        sg.add_nodes_from(reg_list)

        return sg

    def get_max_t2_array(self):
        """ returns volume of min pval per voxel

        Returns:
            max_t2 (array): min pval observed per each voxel
        """
        node_list = self._cut_greedy(node_val_dict=self.node_t2_dict,
                                     max_flag=True)

        max_t2 = np.zeros(self.file_tree.ref.shape)
        for n in node_list:
            mask = self.merge_record.get_pc(node=n).to_mask()
            max_t2[mask] = self.node_t2_dict[n]

        return max_t2
