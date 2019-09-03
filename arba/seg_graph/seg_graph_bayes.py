import numpy as np

import arba.bayes
from .seg_graph import SegGraph
from .seg_graph_hist import SegGraphHistory


class SegGraphBayes(SegGraphHistory):
    """ tracks lower bound of difference among all regions through history

    Attributes:
        alpha (float): confidence in lower bound
        node_bnd_dict (dict): keys are nodes, values are 2-norm of lower bnd
    """

    def __init__(self, *args, alpha=.05, **kwargs):
        super().__init__(*args, obj='t2', **kwargs)
        # init node_t2_dict
        self.alpha = alpha
        self.node_bnd_dict = dict()
        for reg in self.nodes:
            ijk = next(iter(reg.pc_ijk))
            node = self.merge_record.ijk_leaf_dict[ijk]
            self.add(reg, node=node)

    def add(self, reg, node):
        maha = arba.bayes.get_maha(reg)
        self.node_bnd_dict[node] = maha

        return maha

    def merge(self, reg_tuple):
        node_sum = len(self.merge_record.nodes)
        reg_sum = super().merge(reg_tuple)

        # record lower_bnd of newest region
        self.add(reg=reg_sum, node=node_sum)

        return reg_sum

    def cut_greedy_lower_bnd(self):
        """ gets SegGraph of disjoint reg with max lower bnd which covers vol
        """
        # get disjoint cover of entire volume which greedily minimizes pval
        node_list = self._cut_greedy(node_val_dict=self.node_bnd_dict,
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

    def get_max_lower_bnd_array(self):
        """ returns volume of min pval per voxel

        Returns:
            max_lower_bnd (array): min pval observed per each voxel
        """
        node_list = self._cut_greedy(node_val_dict=self.node_bnd_dict,
                                     max_flag=True)

        max_t2 = np.zeros(self.file_tree.ref.shape)
        for n in node_list:
            mask = self.merge_record.get_pc(node=n).to_mask()
            max_t2[mask] = self.node_bnd_dict[n]

        return max_t2
