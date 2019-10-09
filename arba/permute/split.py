import random

import numpy as np


class Split(dict):
    """ characterizes which sbj belongs to which grp

    keys are group labels, values are list of sbj which belong to grp
    """

    @property
    def grp0(self):
        return self.grp_tuple[0]

    @property
    def grp1(self):
        return self.grp_tuple[1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self) == 2, 'Split must have only two groups'

        self.grp_tuple = sorted(self.keys())
        self.sbj_list = self[self.grp0] + self[self.grp1]
        self.num_sbj = len(self.sbj_list)

    def shuffle(self, seed=None):
        """ shuffles sbj between grp, returns new split (same num sbj / grp)"""
        if seed is None:
            return Split(self)

        random.seed(seed)
        sbj_list = list(self.sbj_list)
        random.shuffle(sbj_list)

        n0 = len(self[self.grp0])
        return Split({self.grp0: sbj_list[:n0],
                      self.grp1: sbj_list[n0:]})

    def get_sbj_bool(self, sbj_list):
        assert set(self.sbj_list).issuperset(sbj_list), 'sbj not in split'

        sbj_bool = np.zeros(self.num_sbj).astype(bool)
        for idx, sbj in enumerate(sbj_list):
            if sbj in self[self.grp1]:
                sbj_bool[idx] = True

        return sbj_bool
