from arba.region import *
import pytest


@pytest.fixture
def feat_stat():
    return FeatStat(n=10, mu=[0, 0], cov=np.eye(2))


def test_eq(feat_stat):
    assert feat_stat == feat_stat


def test_add():
    fs0 = FeatStat(n=10, mu=[0, 0], cov=np.zeros((2, 2)))
    fs1 = FeatStat(n=10, mu=[1, 1], cov=np.zeros((2, 2)))

    fs_expected = FeatStat(n=20, mu=[.5, .5], cov=np.ones((2, 2)) * .25)
    assert (fs0 + fs1) == fs_expected
