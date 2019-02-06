from .seg_graph_hist import SegGraphHistory


class SegGraphPerm(SegGraphHistory):
    """ keeps track of max mahalanobis among all regions through history """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_maha = max(reg.maha * len(reg) for reg in self.nodes)

    def combine(self, reg_tuple):
        reg_sum = super().combine(reg_tuple)
        maha = reg_sum.maha * len(reg_sum)
        if maha > self.max_maha:
            self.max_maha = maha
        return reg_sum
