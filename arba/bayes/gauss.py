import numpy as np
import scipy.stats


def get_maha(reg, **kwargs):
    grp_mu_dict, grp_cov_dict = reg.bayes()

    # estimate mu via normal approximation of t distribution
    mu, cov = bayes_mu_delta(grp_mu_dict)

    # get vector closest to origin which has at least
    lower_bnd = get_lower_bnd(mu, cov, **kwargs)

    # average grp covariance
    cov_point_est = np.mean([lam / d for lam, d in grp_cov_dict.values()],
                            axis=0)

    # compute maha
    maha = lower_bnd @ np.linalg.inv(cov_point_est) @ lower_bnd

    return maha


def bayes_mu_delta(grp_mu_dict):
    # order groups
    grp0, grp1 = sorted(grp_mu_dict.keys())

    # get mean covar per grp
    mu0, cov0 = grp_mu_dict[grp0]
    mu1, cov1 = grp_mu_dict[grp1]

    # estimate group difference as gaussian
    delta_mu = mu1 - mu0
    delta_cov = cov0 + cov1

    return delta_mu, delta_cov


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
