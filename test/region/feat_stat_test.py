from arba.region import *

np.random.seed(1)


def rand_fs(n=10, d=2):
    x = np.random.normal(size=(d, n))
    return x, FeatStat.from_array(x)


def test_eq():
    x, fs = rand_fs()
    assert fs == fs, 'equality error'


def test_add():
    fs0 = FeatStat(n=10, mu=np.array([0, 0]), cov=np.zeros((2, 2)))
    fs1 = FeatStat(n=10, mu=np.array([1, 1]), cov=np.zeros((2, 2)))
    fs_sum = FeatStat(n=20, mu=np.array([.5, .5]), cov=np.ones((2, 2)) * .25)

    assert (fs0 + fs1) == fs_sum, 'error add'

    x0, fs0 = rand_fs()
    x1, fs1 = rand_fs()
    fs_sum = FeatStat.from_array(np.hstack((x0, x1)))
    assert (fs0 + fs1) == fs_sum, 'error add random'


def test_scale():
    # original data
    x = np.random.normal(0, 1, size=(3, 10))

    # scaled data
    a = np.random.normal(0, 1, size=(3, 3))

    fs0 = FeatStat.from_array(x)
    fs0 = fs0.scale(a)

    fs1 = FeatStat.from_array(a @ x)
    assert fs1 == fs0, 'error in scale'
