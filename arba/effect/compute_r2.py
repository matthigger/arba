import numpy as np


def compute_r2(beta, y, x, y_pool_cov=None):
    """ computes r2

    Args:
        beta (np.array):
        y (np.array): (num_sbj, dim_img) imaging features
        x (np.array): (num_sbj, dim_sbj) subject features
        y_pool_cov (np.array): pooled (across sbj) of imaging feature cov
                                 across region

    Returns:
        r2 (float): coefficient of determination
    """
    # get tr_pool_cov
    if y_pool_cov is None:
        tr_pool_cov = 0
    else:
        tr_pool_cov = np.trace(y_pool_cov)

    # compute error
    delta = y - x @ beta

    # compute eps, covariance of error
    eps = np.atleast_2d((delta.T @ delta) / delta.shape[0])

    # compute covariance of imaging features
    cov = np.atleast_2d(np.cov(y.T, ddof=0))

    # compute r2
    r2 = 1 - (np.trace(eps) + tr_pool_cov) / (np.trace(cov) + tr_pool_cov)

    return r2