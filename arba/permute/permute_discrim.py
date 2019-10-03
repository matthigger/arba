from .permute import Permute
from ..region import RegionDiscriminate


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
