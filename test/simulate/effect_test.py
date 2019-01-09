import pytest

from pnl_segment.simulate import *


def test_draw_random_u():
    np.random.seed(1)
    x = draw_random_u(3)
    assert np.isclose(np.linalg.norm(x), 1), 'draw_random_u() not on sphere'


@pytest.fixture
def error():
    d = 2
    mean = np.ones(2)
    fs = FeatStat(n=10, mu=np.zeros(d), cov=np.eye(2))
    mask = Mask(np.ones((3, 3)))
    return Effect(mask=mask, mean=mean, fs=fs)

def test_u_maha_setter(error):
    assert np.isclose(error.maha, 2), 'maha computation'

    u_old = error.u
    error.maha = 1
    assert np.isclose(error.maha, 1), 'maha setter'
    assert np.allclose(u_old, error.u), 'u computation error'

    u_target = np.array([-1, 1]).astype(float)
    u_target *= 1 / np.linalg.norm(u_target)
    error.u = u_target
    assert np.allclose(error.u, u_target), 'u setter error'