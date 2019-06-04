import numpy as np
import scipy.stats


def get_gauss_and_min_delta(x, alpha=.05):
    """ estimate prob of delta as a gaussian, returns lower bound on delta

    by lower bound we mean the value of delta which is less probable than (1 -
    alpha) of points (and has min norm on that ellipse)

    Args:
        x (np.array): (num_samples, d) samples drawn from the distribution of
                      delta
        alpha (float): min probability to determine bound

    Returns:
        mv_norm (scipy.stats.multivariate_normal):
        lower_bnd (np.array):
    """

    # build mv normal
    mu = np.mean(x, axis=0)
    cov = np.cov(x.T)
    mv_norm = scipy.stats.multivariate_normal(mean=mu, cov=cov)

    # find mahalanobis distance maha such that 1 - alpha of prob mass has maha
    # less than or equal to it
    maha = scipy.stats.chi2.ppf(1 - alpha, df=len(mu))

    # move from mu towards origin until we reach point whose mahalanobis = maha
    cov_inv = np.linalg.inv(cov)
    offset = -mu
    _maha = offset @ cov_inv @ offset
    lower_bnd = offset * np.sqrt(maha / _maha) + mu
    assert np.isclose(maha,
                      (lower_bnd - mu) @ cov_inv @ (lower_bnd - mu)), \
        'compute error'

    # set lower_bnd to delta=0 is within the 1-alpha most probable points
    if np.linalg.norm(mu) <= np.linalg.norm(mu - lower_bnd):
        lower_bnd = np.zeros(mu.shape)

    return mv_norm, lower_bnd
