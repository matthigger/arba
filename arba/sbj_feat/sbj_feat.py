import numpy as np

from ..permute import get_perm_matrix


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
            self._pseudo_inv = np.linalg.pinv(self.x)
        return self._pseudo_inv

    def get_x_target(self, bias=True):
        if bias:
            return self.x[:, self.contrast]
        else:
            c = np.array(self.contrast)
            c[0] = False
            return self.x[:, c]

    x_target = property(get_x_target)

    def get_x_nuisance(self, bias=True):
        c = np.logical_not(self.contrast)
        c[0] = bias
        return self.x[:, c]

    x_nuisance = property(get_x_nuisance)

    def __init__(self, x, sbj_list=None, feat_list=None, permute_seed=None,
                 contrast=None):
        self.x = None
        self._x = x
        self._pseudo_inv = None
        self.perm_seed = None
        self.perm_matrix = None

        num_sbj, num_feat = x.shape

        # sbj_list
        if sbj_list is None:
            self.sbj_list = [f'sbj{idx}' for idx in range(num_sbj)]
        else:
            assert len(sbj_list) == num_sbj, 'x & sbj_list dimension'
            self.sbj_list = sbj_list

        # feat_list
        if feat_list is None:
            self.feat_list = [f'sbj_feat{idx}' for idx in range(num_feat)]
        else:
            assert len(feat_list) == num_feat, 'x & feat_list dimension'
            self.feat_list = feat_list

        # contrast
        if contrast is None:
            self.contrast = np.ones(self.num_feat).astype(bool)
        else:
            self.contrast = np.array(contrast).astype(bool)
            assert self.contrast.shape == (self.num_feat,), \
                'contrast dimension error'

        # add bias term (constant feature, serves as intercept)
        self._x = np.hstack((np.ones((self.num_sbj, 1)), self._x))
        self.feat_list.insert(0, '1')
        self.contrast = np.insert(self.contrast, 0, True)

        # permute
        self.permute(permute_seed)

    def rm_nuisance(self, beta):
        """ returns a matrix of sbj_feat which adjust for nuisance in beta

        Args:
            beta (np.array): mapping

        Returns:
            x (np.array): (num_sbj, num_feat) features
        """
        raise NotImplementedError

    def permute(self, seed=None):
        """ permutes data according to a random seed

        Args:
            seed: initialization of randomization, associated with a
                          permutation matrix
        """
        if seed == self.perm_seed:
            return

        self.perm_seed = seed
        if self.perm_seed is None:
            self.perm_matrix = None
            self.x = self._x
            return

        self.perm_matrix = get_perm_matrix(self.num_sbj, seed=seed)
        self.x = self.perm_matrix @ self._x

    def freedman_lane(self, img_feat):
        """ shuffles the residuals of img_feat under a nuisance regression

        Args:
            img_feat (np.array): (num_sbj, num_img_feat) imaging features

        Returns:
            img_feat (np.array): (num_sbj, num_img_feat) imaging features with
                                 permuted nuisance residuals
        """

        assert self.is_permuted, 'must be permuted to run freedman_lane()'
        assert img_feat.shape[0] == self.num_sbj, 'num_sbj mismatch'

        # compute nuisance regression
        beta = np.linalg.pinv(self.x_nuisance) @ img_feat

        # compute residuals
        resid = img_feat - beta @ self.x_nuisance

        # subtract residuals, add permuted residuals
        img_feat = img_feat + (self.perm_matrix - np.eye(self.num_sbj)) @ resid

        raise NotImplementedError('not tested')

        return img_feat
