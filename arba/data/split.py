import numpy as np


class Split(dict):
    """ manages partitioning of sbjs, keys are grp labels, values are sbj_list

    Class attributes:
        sbj_list (list): all subjects examined

    Attributes:
        grp_list (list): list of grps (fixes order of groups)
        sbj_grp_dict (dict): keys are sbj, values are grp

    >>> Split.fix_order('abcde')
    >>> split = Split({0: 'abc', 1: 'd', 2: 'e'})
    >>> split.idx_array
    array([0, 0, 0, 1, 2])
    """
    sbj_list = None

    @classmethod
    def fix_order(cls, sbj_list):
        cls.sbj_list = sbj_list

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

    @property
    def idx_array(self):
        idx_array = np.array([self.grp_list.index(self.sbj_grp_dict[sbj])
                              for sbj in self.sbj_list])
        if len(self.grp_list) == 2:
            idx_array = idx_array.astype(bool)

        return idx_array
