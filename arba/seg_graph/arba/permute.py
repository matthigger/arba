from arba.permute import Permute
from ..seg_graph_hist import SegGraphHistory


class PermuteARBA(Permute):
    """ runs ARBA permutations

    Attributes:
        merge_record (MergeRecord): if passed, fixes the hierarchy which is
                                    searched.  if None, a new merge_record is
                                    built each time
        fnc_stat (fnc): accepts Region object, returns statistic of interest
    """

    def __init__(self, *, fnc_stat, merge_record=None, **kwargs):
        super().__init__(**kwargs)
        self.merge_record = merge_record
        self.fnc_stat = fnc_stat

    def run_split(self, *args, **kwargs):
        if self.merge_record is None:
            return self.run_split_new_tree(*args, **kwargs)
        else:
            return self.run_split_fixed(*args, **kwargs)

    def run_split_fixed(self, x, split):
        """ returns a volume of max stat (per vox) across ARBA hierarchy

        Args:
            x (np.array): (space0, space1, space2, num_sbj, num_feat)
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        raise NotImplementedError

    def run_split_new_tree(self, ft, split):
        """ returns a volume of max stat (per vox) across ARBA hierarchy

        Args:
            x (np.array): (space0, space1, space2, num_sbj, num_feat)
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        SegGraphHistory()
