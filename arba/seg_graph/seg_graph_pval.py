import numpy as np

from .seg_graph import SegGraph
from .seg_graph_hist import SegGraphHistory


class SegGraphHistPval(SegGraphHistory):
    """ keeps track of pval among all regions through history

    Attributes:
        node_t2_dict (dict): keys are nodes, values are t2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, obj='t2', **kwargs)
        # init node_pval_dict
        self.node_pval_dict = dict()
        for reg in self.nodes:
            ijk = next(iter(reg.pc_ijk))
            node = self.merge_record.ijk_leaf_dict[ijk]
            self.node_pval_dict[node] = reg.pval

    def merge(self, reg_tuple):
        node_sum = len(self.merge_record.nodes)
        reg_sum = super().merge(reg_tuple)

        # record pval of newest region
        self.node_pval_dict[node_sum] = reg_sum.pval

        return reg_sum

    def cut_greedy_pval(self):
        """ gets SegGraph of disjoint reg with max t2 which covers volume
        """
        # get disjoint cover of entire volume which greedily minimizes pval
        node_list = self._cut_greedy(node_val_dict=self.node_pval_dict,
                                     max_flag=False)

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

    def get_min_pval_array(self):
        """ returns volume of min pval per voxel

        Returns:
            min_pval (array): min pval observed per each voxel
        """
        node_list = self._cut_greedy(node_val_dict=self.node_pval_dict,
                                     max_flag=False)

        min_pval = np.zeros(self.file_tree.ref.shape)
        for n in node_list:
            mask = self.merge_record.get_pc(n).to_mask()
            min_pval[mask] = self.node_pval_dict[n]

        return min_pval
