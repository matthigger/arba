from arba.data import *


def test_data_sbj():
    np.random.seed(1)
    num_sbj, num_feat = 10, 4
    poly_order = 2
    contrast = [1, 1, 0, 0]

    x = np.random.normal(size=(num_sbj, num_feat))

    data_sbj = DataSubject(x)
    data_sbj_w_nuis = DataSubject(x, contrast=contrast)
    data_sbj_p = DataSubjectPoly(x, poly_order=poly_order)
    data_sbj_p_w_nuis = DataSubjectPoly(x, contrast=contrast,
                                        poly_order=poly_order)
