from arba.sbj_feat import *


def test_sbj_feat():
    np.random.seed(1)
    num_sbj, num_feat = 10, 4
    x = np.random.normal(size=(num_sbj, num_feat))
    sbj_feat = SubjectFeatures(x)
