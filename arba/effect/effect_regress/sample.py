import random

import numpy as np

import arba
from .effect_regress import EffectRegress


def get_effect_list(effect_num_vox, shape, num_eff=1, r2=.5, dim_sbj=1,
                    dim_img=1, num_sbj=100, rand_seed=None):
    if rand_seed is not None:
        np.random.seed(1)
        random.seed(1)

    # subject params
    mu_sbj = np.zeros(dim_sbj)
    sig_sbj = np.eye(dim_sbj)

    # imaging params
    mu_img = np.zeros(dim_img)
    sig_img = np.eye(dim_img)

    # dummy reference space
    ref = arba.space.RefSpace(affine=np.eye(4))

    # sample sbj features
    feat_sbj = np.random.multivariate_normal(mean=mu_sbj,
                                             cov=sig_sbj,
                                             size=num_sbj)

    # build feat_img (shape0, shape1, shape2, num_sbj, dim_img)
    feat_img = np.random.multivariate_normal(mean=mu_img,
                                             cov=sig_img,
                                             size=(*shape, num_sbj))

    # build file_tree
    file_tree = arba.data.SynthFileTree.from_array(data=feat_img)

    feat_mapper = arba.regress.FeatMapperStatic(n=dim_sbj,
                                                sbj_list=file_tree.sbj_list,
                                                feat_sbj=feat_sbj)

    # build regression, impose it
    eff_list = list()
    prior_array = np.ones(shape)
    cov_sbj = np.cov(feat_sbj.T, ddof=0)
    with file_tree.loaded():
        for idx in range(num_eff):
            # sample effect extent
            effect_mask = arba.space.sample_mask(prior_array=prior_array,
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

    return feat_sbj, file_tree, eff_list
