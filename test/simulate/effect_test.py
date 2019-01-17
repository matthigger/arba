import pytest

from pnl_segment.simulate import *


def test_draw_random_u():
    np.random.seed(1)
    x = draw_random_u(3)
    assert np.isclose(np.linalg.norm(x), 1), 'draw_random_u() not on sphere'


@pytest.fixture
def effect(d=2, maha=5):
    mean = np.ones(d) * maha * np.sqrt( 1 / d)
    fs = FeatStat(n=10, mu=np.zeros(d), cov=np.eye(d))
    mask = Mask(np.ones((3, 3)))
    effect = Effect(mask=mask, mean=mean, fs=fs)

    assert np.isclose(effect.maha, maha), 'maha computation'

    return effect


def test_u_maha_setter(effect):
    u_old = effect .u
    effect.maha = 1
    assert np.isclose(effect .maha, 1), 'maha setter'
    assert np.allclose(u_old, effect .u), 'u computation error'

    u_target = np.array([-1, 1]).astype(float)
    u_target *= 1 / np.linalg.norm(u_target)
    effect.u = u_target
    assert np.allclose(effect .u, u_target), 'u setter error'
