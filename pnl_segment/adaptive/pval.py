import numpy as np
from scipy.stats import chi2


def mahalanobis2(x, mu, cov=None, cov_inv=None):
    """ computes mahalanobis squared of x
    """

    # get cov_inv
    if (cov is None) == (cov_inv is None):
        raise AttributeError('either cov xor cov_inv required')
    if cov_inv is None:
        cov_inv = np.linalg.inv(cov)

    # compute
    delta = x - mu
    r2 = delta.T @ cov_inv @ delta

    return r2


def get_pval(reg, grp_cmp, grp_test, bessel=True):
    """ returns probability that mean of grp_test's values are from grp_cmp

    assumes each is normally distributed

    Args:
        reg (adaptive.region): region object, contains feat_stat attribute: a
                               dict of feat_stats, labeled by grp
        grp_cmp: comparison group, key of reg.feat_stat to build model from
        grp_test: test group, key of reg.feat_stat to test from model
        bessel (bool): toggles whether bessel correction applied

    Returns:
        pval (float): prob
        r2 (float): mahalanobis squared of mean test to grp_cmp (is chi2)
    """

    # get bessel correction (so cmp is unbiased var)
    # https://en.wikipedia.org/wiki/Bessel%27s_correction
    n_cmp = reg.feat_stat[grp_cmp].n
    bessel = (n_cmp / (n_cmp - 1)) ** bessel

    n_test = reg.feat_stat[grp_cmp].n

    cmp_mu = reg.feat_stat[grp_cmp].mu
    cmp_cov = reg.feat_stat[grp_cmp].cov * bessel / n_test
    test_mu = reg.feat_stat[grp_test].mu

    r2 = mahalanobis2(x=test_mu, mu=cmp_mu, cov=cmp_cov)
    pval = 1 - chi2.cdf(r2, df=len(cmp_mu))

    return float(pval), float(r2)


if __name__ is '__main__':
    # builds plot which suggests mahalanobis squared is chi2 distributed
    # https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf

    import matplotlib.pyplot as plt

    d = 2
    mu = np.zeros(d)
    cov = np.eye(d)
    n = 10000

    cov_inv = np.linalg.inv(cov)
    x = np.random.multivariate_normal(mean=mu, cov=cov, size=n)
    r2_list = [mahalanobis2(_x, mu, cov_inv) for _x in x]

    domain = np.linspace(np.min(r2_list), np.max(r2_list), 200)
    p = chi2.pdf(domain, df=d)

    plt.hist(r2_list, bins=100, normed=True, label='observed mahalanobis^2')
    plt.plot(domain, p, label='prob chi2 expected')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.xlabel('mahalanobis squared')
    plt.ylabel('prob / normed count')
