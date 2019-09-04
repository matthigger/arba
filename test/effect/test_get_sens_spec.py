from arba.effect.get_sens_spec import *


def test_get_sens_spec0(target=[1, 1, 1],
                        estimate=[1, 1, 1],
                        expected_sens_spec=(1, np.nan)):
    sens_spec = get_sens_spec(target=target, estimate=estimate)
    assert expected_sens_spec == sens_spec


def test_get_sens_spec1(target=[1, 1, 0],
                        estimate=[1, 0, 0],
                        expected_sens_spec=(.5, 1)):
    sens_spec = get_sens_spec(target=target, estimate=estimate)
    assert expected_sens_spec == sens_spec


def test_get_sens_spec2(mask=[1, 0, 1],
                        target=[1, 0, 0],
                        estimate=[1, 1, 0],
                        expected_sens_spec=(1, 1)):
    sens_spec = get_sens_spec(target=target, estimate=estimate, mask=mask)
    assert expected_sens_spec == sens_spec
