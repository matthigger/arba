from string import ascii_uppercase as uppercase

import numpy as np

from arba.permute import get_perm_matrix

# reserved name of the 'constant' feature, serves as bias term in regression
bias_feat = '1'


class DataSubject:
    """ contains features which are constant across image (age, sex, ...)

    serves two main functions:
        -encapsulate contrast matrix, which distinguishes 'target' variables
        (those we're clinically interested in) from 'nuisance' variables (those
        whose effects we wish to control for while examining 'target' vars)
        -permutation testing via Freedman Lane (1983).

    Note: we preclude the inclusion of polynomial features which draw from both
    target and nuisance params (e.g. if one is interested in the effects of age
    while controlling for sex, a model with an age * sex term is invalid in
    Freedman Lane)

    Attributes:
        sbj_list (list): list of subjects, fixes their ordering in matrices
        feat_list (list): names of features
        feat (np.array): (num_sbj, num_feat) features
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

    def __init__(self, feat, sbj_list=None, feat_list=None, permute_seed=None,
                 contrast=None, bias=True):
        self.feat = feat
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
            assert bias_feat not in feat_list, \
                f'use of reserved feature name: {bias_feat}'
            self.feat_list = feat_list

        # contrast
        if contrast is None:
            self.contrast = np.ones(self.num_feat).astype(bool)
        else:
            self.contrast = np.array(contrast).astype(bool)
            assert self.contrast.shape == (self.num_feat,), \
                'contrast dimension error'
        self.has_nuisance = np.any(np.logical_not(self.contrast))

        # add bias term (constant feature, serves as intercept)
        if bias:
            self.feat = np.hstack((np.ones((self.num_sbj, 1)), self.feat))
            self.feat_list.insert(0, bias_feat)
            self.contrast = np.insert(self.contrast, 1, True)

        # compute pseudo inverse
        self.pseudo_inv = np.linalg.pinv(self.feat)

        # permute
        self.perm_seed = None
        self.perm_matrix = None
        self.permute(permute_seed)

    def permute(self, seed=None):
        """ permutes data according to a random seed

        Args:
            seed: initialization of randomization, associated with a
                          permutation matrix
        """

        self.perm_seed = seed
        if self.perm_seed is None:
            self.perm_matrix = None
            return

        self.perm_matrix = get_perm_matrix(self.num_sbj, seed=seed)

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

        if not self.has_nuisance:
            # no nuisance params => all feat_img is a 'residual'
            return self.perm_matrix @ feat_img

        # factor feat_img under nuisance params
        effect, residuals = self.factor_nuisance(feat_img)

        # shuffle the residuals
        feat_img = effect + self.perm_matrix @ residuals

        return feat_img

    def factor_nuisance(self, feat_img, beta=None):
        """ splits a feat_img into nuisance effects and residuals

        Args:
            feat_img (np.array): (num_sbj, num_feat_img) imaging features
            beta (np.array): (num_feat_sbj, num_feat_img) mapping

        Returns:
            effect (np.array): (num_sbj, num_feat_img) imaging features driven
                               by nuisance effects
            residuals (np.array): (num_sbj, num_feat_img) imaging features not
                                  caused by nuisance effects
        """
        # compute nuisance regression
        feat_nuisance = self.feat[:, np.logical_not(self.contrast)]
        if beta is None:
            beta = np.linalg.pinv(feat_nuisance) @ feat_img
        else:
            # only take nuisance portion of beta
            beta = beta[feat_nuisance, :]

        # factor
        effect = feat_nuisance @ beta
        residuals = feat_img - effect

        return effect, residuals

    def linspace(self, idx, n=100, feat=None):
        """ returns feat_img which spans the range of values

        Args:
            idx (int): index of feature to change
            n (int): number of observations to return
            feat (np.array): (num_obs, num_feat): subject features (defaults to
                                                  self.feat)

        Returns:
             feat (np.array): (n, self.num_feat) all features are their
                              observation average, except for the idx-th
                              feature which is n linearly spaced points from
                              min to max
        """
        if feat is None:
            feat = self.feat

        feat_img = feat.mean(axis=0)
        feat_img = np.repeat(np.atleast_2d(feat_img), repeats=n, axis=0)
        feat_img[:, idx] = np.linspace(min(feat[:, idx]),
                                       max(feat[:, idx]), n)
        return feat_img
