import numpy as np


class SubjectFeatures:
    """ contains features which are constant across image (age, sex, ...)

    serves two main functions:
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
        feat_list (list): names of features
        x (np.array): (num_sbj, num_feat) features (possibly permuted)
        _x (np.array): (num_sbj, num_feat) features (never permuted)
        permute_seed: if None, features are unpermuted.  otherwise denotes the
                      seed which generated the permutation matrix applied
        permute_matrix (np.array): (num_feat, num_feat) permutation matrix
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

    def __init__(self, x, sbj_list=None, feat_list=None, permute_seed=None,
                 contrast=None):
        self.x = None
        self._x = x
        self.permute_seed = None
        self.permute_matrix = None

        num_sbj, num_feat_raw = x.shape

        # sbj_list
        if sbj_list is None:
            self.sbj_list = [f'sbj{idx}' for idx in enumerate(num_sbj)]
        else:
            assert len(sbj_list) == num_sbj, 'x & sbj_list dimension'
            self.sbj_list = sbj_list

        # feat_list
        if feat_list is None:
            self.feat_list = [f'sbj_feat{idx}' for idx in enumerate]
        else:
            assert len(feat_list) == num_feat_raw, 'x & feat_list dimension'
            self.feat_list = feat_list

        # contrast
        if contrast is None:
            self.contrast = np.ones(self.num_feat)
        else:
            self.contrast = np.array(contrast).astype(bool)
            assert np.array_equal(self.contrast.shape, self.num_feat), \
                'contrast dimension error'

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
