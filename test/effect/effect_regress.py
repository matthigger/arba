from math import isclose

import numpy as np

from arba.effect import compute_r2, EffectRegress


def feat_sbj_img_beta(num_sbj=5, dim_sbj=3, dim_img=1, seed=1):
    np.random.seed(seed)
    feat_sbj = np.random.normal(size=(num_sbj, dim_sbj))
    img_feat = np.random.normal(size=(num_sbj, dim_img))

    beta = np.linalg.pinv(feat_sbj) @ img_feat

    return feat_sbj, img_feat, beta


def test_compute_r2(feat_sbj_img_beta):
    feat_sbj, img_feat, beta = feat_sbj_img_beta()
    compute_r2(beta=beta, y=img_feat, x=feat_sbj)


def test_effect_regress():
    feat_sbj, img_feat, beta = feat_sbj_img_beta()
    mask = np.eye(3)

    eff = EffectRegress(mask=mask, beta=beta)

    for r2 in [.1, .5, .9]:
        eff = EffectRegress.from_r2(r2=r2, img_feat=img_feat,
                                    feat_sbj=feat_sbj, mask=mask)
        img_feat_adjust = img_feat + feat_sbj @ eff.beta
        r2_observed = compute_r2(beta=eff.beta, y=img_feat_adjust, x=feat_sbj)
        assert isclose(r2_observed, r2, rel_tol=1e-5), 'r2 compute error'
