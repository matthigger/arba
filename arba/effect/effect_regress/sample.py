import random

import numpy as np
from scipy.ndimage.morphology import binary_erosion

import arba
from arba.space import sample_mask_min_var, sample_mask
from .effect_regress import EffectRegress


def get_effect_list(effect_num_vox, file_tree, num_eff=1, r2=.5, dim_sbj=1,
                    rand_seed=None, no_edge=False, min_var_mask=False):
    if rand_seed is not None:
        np.random.seed(1)
        random.seed(1)

    # subject params
    mu_sbj = np.zeros(dim_sbj)
    sig_sbj = np.eye(dim_sbj)

    # dummy reference space
    ref = arba.space.RefSpace(affine=np.eye(4))

    # sample sbj features
    feat_sbj = np.random.multivariate_normal(mean=mu_sbj,
                                             cov=sig_sbj,
                                             size=file_tree.num_sbj)

    feat_mapper = arba.regress.FeatMapperStatic(n=dim_sbj,
                                                sbj_list=file_tree.sbj_list,
                                                feat_sbj=feat_sbj)

    # build regression, impose it
    eff_list = list()
    prior_array = file_tree.mask
    if no_edge:
        prior_array = binary_erosion(prior_array)
    cov_sbj = np.cov(feat_sbj.T, ddof=0)
    with file_tree.loaded():
        for idx in range(num_eff):
            # sample effect extent
            if min_var_mask:
                effect_mask = sample_mask_min_var(num_vox=effect_num_vox,
                                                  file_tree=file_tree,
                                                  prior_array=prior_array)
            else:
                effect_mask = sample_mask(prior_array=prior_array,
                                          num_vox=effect_num_vox,
                                          ref=ref)

            # get imaging feature stats in this mask
            fs = file_tree.get_fs(mask=effect_mask)

            # construct effect
            eff = EffectRegress.from_r2(r2=r2,
                                        mask=effect_mask,
                                        eps_img=fs.cov,
                                        cov_sbj=cov_sbj,
                                        feat_mapper=feat_mapper)
            eff_list.append(eff)

    return feat_sbj, eff_list
