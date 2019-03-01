import pytest

from arba.simulate import *


def test_draw_random_u():
    np.random.seed(1)
    x = draw_random_u(3)
    assert np.isclose(np.linalg.norm(x), 1), 'draw_random_u() not on sphere'


@pytest.fixture
def effect(d=2, t2=5):
    mean = np.ones(d) * t2 * np.sqrt(1 / d)
    fs = FeatStat(n=10, mu=np.zeros(d), cov=np.eye(d))
    mask = Mask(np.ones((3, 3)))
    effect = Effect(mask=mask, mean=mean, fs=fs)

    assert np.isclose(effect.t2, t2), 't2 computation'

    return effect


def test_u_t2_setter(effect):
    u_old = effect .u
    effect.t2 = 1
    assert np.isclose(effect .t2, 1), 't2 setter'
    assert np.allclose(u_old, effect .u), 'u computation error'

    u_target = np.array([-1, 1]).astype(float)
    u_target *= 1 / np.linalg.norm(u_target)
    effect.u = u_target
    assert np.allclose(effect .u, u_target), 'u setter error'
