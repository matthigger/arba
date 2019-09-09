import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .sbj_feat import SubjectFeatures


class SubjectFeaturesPoly(SubjectFeatures):
    """ polynomial projection of subject features for non-linear regression

    Note: we preclude the inclusion of polynomial features which draw from both
    target and nuisance params (e.g. if one is interested in the effects of age
    while controlling for sex, a model with an age * sex term makes Freedman
    Lane difficult, maybe impossible)

    Attributes:
        poly_order (int): order of polynomial projection
        feat_list_raw (list): names of raw features (before projection)
        self.contrast_raw (np.array): which raw features are targets
        x_raw (np.array): (num_sbj, num_feat) original features, unpermuted
    """

    def __init__(self, *args, poly_order=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.poly_order = poly_order
        self.feat_list_raw = self.feat_list
        self.contrast_raw = self.contrast
        self.x_raw = self._x

        # project target features
        poly = PolynomialFeatures(degree=poly_order, include_bias=True)
        x_target = poly.fit_transform(self.x_target)
        feat_target = [f for (f, c) in zip(self.feat_list, self.contrast) if c]
        self.feat_list = poly.get_feature_names(feat_target)

        if any(np.logical_not(self.contrast)):
            # project nuisance features
            poly = PolynomialFeatures(degree=poly_order, include_bias=False)
            x_nuisance = poly.fit_transform(self.x_nuisance)
            feat_nuis = [f for (f, c) in zip(self.feat_list, self.contrast) if
                         ~c]
            self.feat_list += poly.get_feature_names(feat_nuis)
            self._x = np.hstack((x_target, x_nuisance))

            # build new contrast matrix
            self.contrast = np.array([1] * x_target.shape[1] +
                                     [0] * x_nuisance.shape[1])
        else:
            # no nuisance features
            self._x = x_target
            self.contrast = np.ones(self._x.shape[1])

        # permute
        self.permute(self.perm_seed)
