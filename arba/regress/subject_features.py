import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class SubjectFeatures:
    """ contains features which are constant across image (age, sex, ...)

    serves three main functions:
        -polynomial projections of subject features
        -encapsulate contrast matrix, which distinguishes 'target' variables
        (those we're clinically interested in) from 'nuisance' variables (those
        whose effects we wish to control for while examining 'target' vars)
        -permutation testing via Freedman Lane

    Note: we preclude the inclusion of polynomial features which draw from both
    target and nuisance params (e.g. if one is interested in the effects of age
    while controlling for sex, a model with an age * sex term makes Freedman
    Lane difficult, maybe impossible)

    Attributes:
        sbj_list (list): list of subjects, fixes their ordering in matrices
        poly_order (int): order of polynomial projection
        feat_list_raw (list): names of raw features (pre projection)
        feat_list (list): names of features (after projection)
        x (np.array): (num_sbj, num_feat) projected features (after projection)
        permute_seed: if None, features are unpermuted.  otherwise denotes the
                      seed which generated the permutation matrix applied
        permute_matrix (np.array): (num_feat, num_feat) permutation matrix
        _x (np.array): (num_sbj, num_feat) project features (unpermuted)
        contrast (np.array): (num_feat) boolean vector.  note that unlike
                             randomise, there are no negative values
    """

    @property
    def num_sbj(self):
        return len(self.sbj_list)

    @property
    def num_feat(self):
        return len(self.feat_list)

    @property
    def is_permuted(self):
        return self.permute_seed is not None

    @property
    def x_target(self):
        return self.x[:, self.contrast]

    @property
    def x_nuisance(self):
        return self.x[:, ~self.contrast]

    def __init__(self, x, poly_order=1, sbj_list=None, feat_list=None,
                 permute_seed=None, contrast=None):
        num_sbj, num_feat_raw = x.shape

        # placeholder to init these in constructor
        self.x = None
        self.permute_seed = None
        self.permute_matrix = None

        # sbj_list
        if sbj_list is None:
            self.sbj_list = [f'sbj{idx}' for idx in enumerate(num_sbj)]
        else:
            assert len(sbj_list) == num_sbj, 'x & sbj_list dimension'
            self.sbj_list = sbj_list

        # feat_list_raw
        if feat_list is None:
            self.feat_list_raw = [f'sbj_feat{idx}' for idx in enumerate]
        else:
            assert len(feat_list) == num_feat_raw, 'x & feat_list dimension'
            self.feat_list_raw = feat_list

        # contrast
        if contrast is None:
            self.contrast = np.ones(self.num_feat)
        else:
            self.contrast = np.array(contrast).astype(bool)
            assert np.array_equal(self.contrast.shape, self.num_feat), \
                'contrast dimension error'

        # project features
        self.poly_order = poly_order

        poly = PolynomialFeatures(degree=poly_order, include_bias=True)
        x_target = poly.fit_transform(x[:, self.contrast])
        feat_target = [f for (f, c) in zip(self.feat_list, self.contrast) if c]
        self.feat_list = poly.get_feature_names(feat_target)

        poly = PolynomialFeatures(degree=poly_order, include_bias=False)
        x_nuisance = poly.fit_transform(x[:, self.contrast])
        feat_nuis = [f for (f, c) in zip(self.feat_list, self.contrast) if ~c]
        self.feat_list += poly.get_feature_names(feat_nuis)

        self.x = np.vstack((x_target, x_nuisance))

        # permute
        self.permute(permute_seed)

    def permute(self, permute_seed=None):
        """ permutes data according to a random seed

        Args:
            permute_seed: initialization of randomization, associated with a
                          permutation matrix
        """
        self.permute_seed = permute_seed
        if self.permute_seed is None:
            self.permute_matrix = None
            self.x = self._x
            return

        raise NotImplementedError

    def freedman_lane(self, img_feat):
        """ shuffles the residuals of img_feat under a nuisance regression

        Args:
            img_feat (np.array): (num_sbj, num_img_feat) imaging features

        Returns:
            img_feat (np.array): (num_sbj, num_img_feat) imaging features with
                                 permuted nuisance residuals
        """

        # compute nuisance regression

        # compute residuals

        # subtract residuals, add permuted residuals

        pass
