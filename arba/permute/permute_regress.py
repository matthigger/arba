from .permute import Permute
from ..region import RegionRegress


class PermuteRegress(Permute):
    """ runs permutation testing to find regions whose r^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        data_sbj (DataSubject): observed subject data
    """
    stat = 'r2'
    reg_cls = RegionRegress

    def __init__(self, data_sbj=None, *args, **kwargs):
        assert data_sbj is not None, 'data_sbj required'
        self.data_sbj = data_sbj
        super().__init__(*args, **kwargs)

    def _set_seed(self, seed=None):
        self.data_sbj.permute(seed)
        RegionRegress.set_data_sbj(self.data_sbj)
