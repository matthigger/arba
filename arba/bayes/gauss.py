import numpy as np
import scipy.stats


def get_lower_bnd(mu, cov, alpha=.05):
    """ estimate prob of delta as a gaussian, returns lower bound on delta

    by lower bound we mean the value of delta which is less probable than (1 -
    alpha) of points (and has min norm on that ellipse)

    Args:
        mu (np.array): mean
        cov (np.array): cov
        alpha (flaot): confidence level

    Returns:
        lower_bnd (np.array):
    """
    # ensure the proper shape
    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    # find mahalanobis distance maha such that 1 - alpha of prob mass has maha
    # less than or equal to it
    maha = scipy.stats.chi2.ppf(1 - alpha, df=len(mu))

    # move from mu towards origin until we reach point whose mahalanobis = maha
    cov_inv = np.linalg.inv(cov)
    offset = -mu
    _maha = offset @ cov_inv @ offset
    offset *= np.sqrt(maha / _maha)
    lower_bnd = offset + mu
    assert np.isclose(maha, offset @ cov_inv @ offset), 'compute error'

    # is the origin in the ellipse which contains 95% of the prob?
    if np.linalg.norm(mu) <= np.linalg.norm(mu - lower_bnd):
        # if so, then the lower bound is 0
        lower_bnd = np.zeros(mu.shape)

    return lower_bnd
