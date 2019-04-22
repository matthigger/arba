from arba.simulate.tfce import apply_tfce
from ...permute_base import PermuteBase


class PermuteTFCE(PermuteBase):
    def __init__(self, *args, h=2, e=.5, c=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.e = e
        self.c = c

    def run_split(self, split, **kwargs):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        t2 = self.get_t2(split)
        t2_tfce = apply_tfce(t2, h=self.h, e=self.e, c=self.c)
        return t2_tfce
