from collections import defaultdict

import numpy as np
import random

class Split(dict):
    """ manages partitioning of sbjs, keys are grp labels, values are sbj_list

    Class attributes:
        sbj_list (list): all subjects examined

    Attributes:
        grp_list (list): list of grps (fixes order of groups)
        sbj_grp_dict (dict): keys are sbj, values are grp

    >>> Split.fix_order('abcde')
    >>> split = Split({'grp0': 'abc', 'grp1': 'd', 'grp2': 'e'})
    >>> split.as_tuple
    ('grp0', 'grp0', 'grp0', 'grp1', 'grp2')
    >>> random.seed(1)
    >>> split.sample()
    {'grp0': ['c', 'd', 'e'], 'grp1': ['a'], 'grp2': ['b']}
    """
    sbj_list = None

    @classmethod
    def fix_order(cls, sbj_list):
        cls.sbj_list = list(sbj_list)

    def from_tuple(self, split_tuple):
        assert len(split_tuple) == len(self.sbj_list), 'invalid length'

        d = defaultdict(list)
        for sbj, grp in zip(self.sbj_list, split_tuple):
            d[grp].append(sbj)

        return Split(d)

    def __init__(self, *args, **kwargs):
        if Split.sbj_list is None:
            raise AttributeError('call Split.fix_order before building Split')

        super().__init__(*args, **kwargs)

        assert len(Split.sbj_list) == sum(len(l) for l in self.values()), \
            'doesnt partition sbj_list'

        self.grp_list = sorted(self.keys())

        self.sbj_grp_dict = dict()
        for grp, sbj_list in self.items():
            for sbj in sbj_list:
                self.sbj_grp_dict[sbj] = grp

    def __len__(self):
        return len(self.sbj_list)

    def sample(self):
        """ returns a permuted split

        note: each split has same number of subjects per group

        Returns:
            split_list (list): list of splits
        """

        d = defaultdict(list)
        sbj_list = list(self.sbj_list)
        random.shuffle(sbj_list)
        assert sbj_list != self.sbj_list, 'original indexing lost'
        for sbj, grp in zip(sbj_list,
                            self.as_tuple):
            d[grp].append(sbj)

        return Split(d)

    def get_bool(self, grp, negate=False):
        sbj_bool = np.array([sbj in self[grp] for sbj in self.sbj_list])

        if negate:
            sbj_bool = np.logical_not(sbj_bool)

        return sbj_bool

    def bool_iter(self):
        for grp in self.grp_list:
            yield grp, self.get_bool(grp)

    @property
    def as_tuple(self):
        return tuple(self.sbj_grp_dict[sbj] for sbj in self.sbj_list)

    def __hash__(self):
        return hash(self.as_tuple)

    def __eq__(self, other):
        if self.grp_list != other.grp_list:
            return False
        return self.as_tuple == other.as_tuple
