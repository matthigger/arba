from string import ascii_uppercase as uppercase

import numpy as np

from arba.permute import get_perm_matrix


class DataSubject:
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
        feat (np.array): (num_sbj, num_feat) features (possibly permuted)
        _feat (np.array): (num_sbj, num_feat) features (never permuted)
        permute_seed: if None, features are unpermuted.  otherwise denotes the
                      seed which generated the permutation matrix applied
        perm_matrix (np.array): (num_feat, num_feat) permutation matrix
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
        return self.perm_seed is not None

    @property
    def pseudo_inv(self):
        # memoize
        if self._pseudo_inv is None:
            self._pseudo_inv = np.linalg.pinv(self.feat)
        return self._pseudo_inv

    def get_x_target(self, bias=True):
        if bias:
            return self.feat[:, self.contrast]
        else:
            c = np.array(self.contrast)
            c[0] = False
            return self.feat[:, c]

    x_target = property(get_x_target)

    def get_x_nuisance(self, bias=True):
        c = np.logical_not(self.contrast)
        c[0] = bias
        return self.feat[:, c]

    x_nuisance = property(get_x_nuisance)

    def __init__(self, feat, sbj_list=None, feat_list=None, permute_seed=None,
                 contrast=None):
        self.feat = None
        self._feat = feat
        self._pseudo_inv = None
        self.perm_seed = None
        self.perm_matrix = None

        num_sbj, num_feat = feat.shape

        # sbj_list
        if sbj_list is None:
            self.sbj_list = [f'sbj{idx}' for idx in range(num_sbj)]
        else:
            assert len(sbj_list) == num_sbj, 'feat & sbj_list dimension'
            self.sbj_list = sbj_list

        # feat_list
        if feat_list is None:
            self.feat_list = [f'feat_sbj{c}' for c in uppercase[:num_feat]]
        else:
            assert len(feat_list) == num_feat, 'feat & feat_list dimension'
            self.feat_list = feat_list

        # contrast
        if contrast is None:
            self.contrast = np.ones(self.num_feat).astype(bool)
        else:
            self.contrast = np.array(contrast).astype(bool)
            assert self.contrast.shape == (self.num_feat,), \
                'contrast dimension error'

        # add bias term (constant feature, serves as intercept)
        self._feat = np.hstack((np.ones((self.num_sbj, 1)), self._feat))
        self.feat_list.insert(0, '1')
        self.contrast = np.insert(self.contrast, 0, True)

        # permute
        self.permute(permute_seed)

    def permute(self, seed=None):
        """ permutes data according to a random seed

        Args:
            seed: initialization of randomization, associated with a
                          permutation matrix
        """
        self._pseudo_inv = None

        self.perm_seed = seed
        if self.perm_seed is None:
            self.perm_matrix = None
            self.feat = self._feat
            return

        self.perm_matrix = get_perm_matrix(self.num_sbj, seed=seed)
        self.feat = self.perm_matrix @ self._feat

    def freedman_lane(self, feat_img):
        """ shuffles the residuals of feat_img under a nuisance regression

        Args:
            feat_img (np.array): (num_sbj, num_feat_img) imaging features

        Returns:
            feat_img (np.array): (num_sbj, num_feat_img) imaging features with
                                 permuted nuisance residuals
        """

        assert self.is_permuted, 'must be permuted to run freedman_lane()'
        assert feat_img.shape[0] == self.num_sbj, 'num_sbj mismatch'

        # compute nuisance regression
        beta = np.linalg.pinv(self.x_nuisance) @ feat_img

        # compute residuals
        resid = feat_img - beta @ self.x_nuisance

        # subtract residuals, add permuted residuals
        feat_img = feat_img + (self.perm_matrix - np.eye(self.num_sbj)) @ resid

        raise NotImplementedError('not tested')

        return feat_img
