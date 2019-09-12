from arba.data import *


def test_data_sbj():
    np.random.seed(1)
    num_sbj, num_feat = 10, 4
    poly_order = 2
    contrast = [1, 1, 0, 0]

    x = np.random.normal(size=(num_sbj, num_feat))

    sbj_feat = DataSubject(x)
    sbj_feat_w_nuis = DataSubject(x, contrast=contrast)
    sbj_feat_p = DataSubjectPoly(x, poly_order=poly_order)
    sbj_feat_p_w_nuis = DataSubjectPoly(x, contrast=contrast,
                                        poly_order=poly_order)
