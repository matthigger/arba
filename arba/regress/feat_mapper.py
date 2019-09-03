import numpy as np


class FeatMapper:
    """ maps features for into domain of regression fnc

    Attributes:
        feat_list (list): names of features

    in addition to insulating the regression (matrix / vector lin algebra) from
    the data sources (dictionaries with clinical meaning) the feature mapper
    encapsulates two functions:
    1) adds a constant feature (always 1, always the first feature) which
    serves as the 'y intercept' of the regression
    2) supports additional basis functions (not implemented yet ...)
    """

    # reserved name of constant feature
    CONSTANT = 'constant'

    @property
    def dim(self):
        return len(self.feat_list)

    def __init__(self, constant_flag=True, feat_list=None, n=None):
        assert (feat_list is None) != (n is None), \
            'either feat_list or n required'

        self.feat_list = list()
        if constant_flag:
            self.feat_list.append(self.CONSTANT)

        if n is not None:
            feat_list = [f'sbj_feat_{idx}' for idx in range(n)]

        self.feat_list += feat_list

    def __call__(self, x, multi=False):
        if multi:
            # x is an iterator of elements, each of which has a feature vector.
            # return (num_element x num_feat) matrix
            return np.hstack(self(_x) for _x in x)

        return self._call_single(x)

    def _call_single(self, x):
        feat_vec = np.empty(shape=self.dim)
        for idx, feat in self.feat_list:
            if feat == self.CONSTANT:
                feat_vec[idx] = 1
            else:
                feat_vec[idx] = x[feat]

        return feat_vec
