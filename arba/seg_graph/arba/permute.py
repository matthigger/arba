from arba.permute import Permute
from arba.plot import save_fig, size_v_wt2, size_v_t2
from ..seg_graph_hist import SegGraphHistory
from ..seg_graph_t2 import SegGraphT2


class PermuteARBA(Permute):
    """ runs ARBA permutations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sg_hist = None

    def save(self, folder, split, print_tree=True, **kwargs):
        """ saves output images in a folder"""
        super().save(folder=folder, split=split, **kwargs)

        if print_tree:
            assert self.sg_hist is not None, 'call determine_sig() first'

            merge_record = self.sg_hist.merge_record
            # todo: better way to store effect
            effect = self.file_tree.split_effect[1]

            sbj_bool_to_list = self.file_tree.sbj_bool_to_list
            not_split = tuple(not (x) for x in split)
            grp_sbj_dict = {'ctrl': sbj_bool_to_list(not_split),
                            'effect': sbj_bool_to_list(split)}

            tree_hist, \
            node_reg_dict = merge_record.resolve_hist(self.file_tree,
                                                      grp_sbj_dict)
            size_v_wt2(tree_hist, mask=effect.mask)
            save_fig(f_out=folder / 'size_v_wt2.pdf')

            size_v_t2(tree_hist, mask=effect.mask)
            save_fig(f_out=folder / 'size_v_t2.pdf')

    def _split_to_sg_hist(self, split, full_t2=False):
        """ builds sg_hist from a split

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split
            full_t2 (bool): toggles instantiating SegGraphT2, a subclass of
                            SegGraphHistory which tracks t2 per each node.

        Returns:
            sg_hist (SegGraphHistory): reduced as much as possible
        """
        not_split = tuple(not x for x in split)
        grp_sbj_dict = {'0': self.file_tree.sbj_bool_to_list(split),
                        '1': self.file_tree.sbj_bool_to_list(not_split)}
        if full_t2:
            sg_hist = SegGraphT2(file_tree=self.file_tree,
                                 grp_sbj_dict=grp_sbj_dict)
        else:
            sg_hist = SegGraphHistory(file_tree=self.file_tree,
                                      grp_sbj_dict=grp_sbj_dict)
        return sg_hist

    def run_split(self, split, **kwargs):
        """ returns max stat (per vox) across new ARBA hierarchy

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this splits

        Returns:
            sg_hist (SegGraphHistory): reduced as much as possible
        """
        sg_hist = self._split_to_sg_hist(split, **kwargs)
        sg_hist.reduce_to(1)

        return sg_hist

    def run_split_max(self, split, **kwargs):
        return split, self.run_split(split).max_t2

    def determine_sig(self, split=None, stat_volume=None):
        """ runs on the original case, uses the stats saved to determine sig"""

        # get volume of stat
        sg_hist = self.run_split(split, full_t2=True)
        max_t2 = sg_hist.get_max_t2_array()

        # store sg_hist of split
        self.sg_hist = sg_hist

        return super().determine_sig(stat_volume=max_t2)
