from .permute import Permute
from ..region import RegionDiscriminate
from ..seg_graph import SegGraphHistory


def get_t2(reg, **kwargs):
    return reg.t2


class PermuteDiscriminate(Permute):
    """ runs permutation testing to find regions whose t^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        split (Split): keys are grps, values are list of sbj
    """

    stat = 't2'
    reg_cls = RegionDiscriminate

    def __init__(self, *args, split, **kwargs):
        self.split = split
        super().__init__(*args, **kwargs)

    def _set_seed(self, seed=None):
        split = self.split.shuffle(seed=seed)
        RegionDiscriminate.set_split(split)

    def get_sg_hist(self, seed=None):
        self._set_seed(seed)
        return SegGraphHistory(data_img=self.data_img,
                               cls_reg=RegionDiscriminate,
                               fnc_dict={'t2': get_t2})
