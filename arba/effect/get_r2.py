import numpy as np


def get_r2(beta, y, x, y_pool_cov=None, contrast=None):
    """ computes r2

    Args:
        beta (np.array): (dim_x, dim_y) linear mapping
        x (np.array): (num_obs, dim_x) features in the mapping's domain
        y (np.array): (num_obs, dim_y) features in mapping's range
        y_pool_cov (np.array): pooled covariance of x (per observation)
        contrast (np.array): (dim_x) boolean array, describes whether each x
                             feature is a target variable (True) or nuisance
                             (False).  all nuisance effects in a given beta are
                             subtracted before r2 is computed

    Returns:
        r2 (float): coefficient of determination
    """
    if contrast is not None:
        # subtract effect of nuisance variables from y
        flag_nuisance = np.logical_not(contrast)
        y = y - x[:, flag_nuisance] @ beta[flag_nuisance, :]

        # zero nuisance feature (after copying to preserve input x)
        x = np.array(x)
        x[:, flag_nuisance] = 0

    # get tr_pool_cov
    if y_pool_cov is None:
        tr_pool_cov = 0
    else:
        tr_pool_cov = np.trace(y_pool_cov)

    # compute error
    error = y - x @ beta

    # compute mse
    mse = np.atleast_2d((error.T @ error) / error.shape[0])

    # compute covariance of imaging features
    y_cov = np.atleast_2d(np.cov(y.T, ddof=0))

    # compute r2
    r2 = 1 - (np.trace(mse) + tr_pool_cov) / (np.trace(y_cov) + tr_pool_cov)

    return r2