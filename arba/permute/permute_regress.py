from .permute import Permute
from ..region import RegionRegress
from ..seg_graph import SegGraphHistory


def get_r2(reg, **kwargs):
    return reg.r2


class PermuteRegress(Permute):
    """ runs permutation testing to find regions whose r^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        data_sbj (DataSubject): observed subject data
    """
    stat = 'r2'
    reg_cls = RegionRegress

    def __init__(self, data_sbj, *args, **kwargs):
        self.data_sbj = data_sbj
        super().__init__(*args, **kwargs)

    def _set_seed(self, seed=None):
        self.data_sbj.permute(seed)
        RegionRegress.set_data_sbj(self.data_sbj)

    def get_sg_hist(self, seed=None):
        self._set_seed(seed)
        return SegGraphHistory(data_img=self.data_img,
                               cls_reg=RegionRegress,
                               fnc_dict={'r2': get_r2})
