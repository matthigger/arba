from arba.region import *

np.random.seed(1)


def rand_fs(n=10, d=2):
    x = np.random.normal(size=(d, n))
    return FeatStat.from_array(x)


def test_eq(fs=rand_fs()):
    assert fs == fs, 'equality error'


def test_add_sub(fs0=FeatStat(n=10, mu=[0, 0], cov=np.zeros((2, 2))),
                 fs1=FeatStat(n=10, mu=[1, 1], cov=np.zeros((2, 2)))):
    fs_sum = FeatStat(n=20, mu=[.5, .5], cov=np.ones((2, 2)) * .25)

    assert (fs0 + fs1) == fs_sum, 'error add'
    assert (fs_sum - fs0) == fs1, 'error subtract'


def test_add_sub_rand():
    fs0 = rand_fs()
    fs1 = rand_fs()
    assert (fs0 + fs1) - fs1 == fs0, 'error: add or subtract in random'
