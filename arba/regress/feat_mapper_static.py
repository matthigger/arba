from .feat_mapper import FeatMapper


class FeatMapperStatic(FeatMapper):
    """ serves as lookup table of sbj features (via sbj)

    Attributes:
        sbj_list (list): list of subjects
        feat_sbj (np.array): features per sbj
    """

    def __init__(self, sbj_list, feat_sbj, *args, constant_flag=False,
                 **kwargs):
        assert constant_flag is False, 'constant_flag invalid'

        super().__init__(*args, constant_flag=constant_flag, **kwargs)

        self.sbj_list = sbj_list
        self.feat_sbj = feat_sbj

    def _call_single(self, x):
        sbj_idx = self.sbj_list.index(x)
        return self.feat_sbj[sbj_idx, :]
