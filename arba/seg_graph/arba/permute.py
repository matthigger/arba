from arba.permute import Permute
from ..seg_graph_hist import SegGraphHistory
from ..seg_graph_t2 import SegGraphT2


class PermuteARBA(Permute):
    """ runs ARBA permutations
    """

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
        not_split = (~x for x in split)
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

        return super().determine_sig(stat_volume=max_t2)
