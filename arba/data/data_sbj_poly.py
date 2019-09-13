import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .data_sbj import DataSubject, bias_feat


class DataSubjectPoly(DataSubject):
    """ polynomial projection of subject features for non-linear regression

    Note: we preclude the inclusion of polynomial features which draw from both
    target and nuisance params (e.g. if one is interested in the effects of age
    while controlling for sex, a model with an age * sex term makes Freedman
    Lane difficult, maybe impossible)

    Attributes:
        poly_order (int): order of polynomial projection
        feat_list_raw (list): names of raw features (before projection)
        contrast_raw (np.array): (num_feat) which raw features are targets
        feat_raw (np.array): (num_sbj, num_feat) original features, unpermuted
    """

    def __init__(self, *args, poly_order=1, interaction_only=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.poly_order = poly_order
        self.feat_raw = self.feat
        self.feat_list_raw = self.feat_list
        self.contrast_raw = self.contrast

        # strip bias_feat (will be added later via PolynomialFeatures)
        if bias_feat in self.feat_list_raw:
            idx = self.feat_list_raw.index(bias_feat)
            self.feat = np.delete(self.feat, idx, axis=1)
            self.feat_list.pop(idx)
            self.contrast = np.delete(self.contrast, idx, axis=0)

        # get idx of target and nuisance features (bias_feat is neither)
        idx_target = np.where(self.contrast)[0]
        idx_nuisance = np.where(np.logical_not(self.contrast))[0]

        # get lists of feature names
        feat_list_trgt = [self.feat_list[idx] for idx in idx_target]
        feat_list_nuis = [self.feat_list[idx] for idx in idx_nuisance]

        # project target features
        poly = PolynomialFeatures(degree=poly_order, include_bias=True,
                                  interaction_only=interaction_only)
        self.feat = poly.fit_transform(self.feat[:, idx_target])
        self.feat_list = poly.get_feature_names(feat_list_trgt)
        self.contrast = np.ones(self.feat.shape[1]).astype(bool)

        if any(np.logical_not(self.contrast)):
            # project nuisance features
            poly = PolynomialFeatures(degree=poly_order, include_bias=False,
                                      interaction_only=interaction_only)
            feat_nuis = poly.fit_transform(self.feat[:, idx_nuisance])
            self.feat_list += poly.get_feature_names(feat_list_nuis)
            self.feat = np.hstack((self.feat, feat_nuis))

            # append nuisance to new contrast matrix
            contrast_nuis = np.zeros(feat_nuis.shape[1]).astype(bool)
            self.contrast = np.stack(self.contrast, contrast_nuis)

        # compute pseudo inverse
        self.pseudo_inv = np.linalg.pinv(self.feat)

        # permute
        self.permute(self.perm_seed)
