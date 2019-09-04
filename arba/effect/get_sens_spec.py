import numpy as np


def get_sens_spec(target, estimate, mask=None):
    """ returns sens + spec
    
    Args:
        target (np.array): location to be estimated
        estimate (np.array): location estimate
        mask (np.array): location in which sensitivity / specificity is counted
                         (defaults to all locations)
    Returns:
        sens (float): percentage of target voxels detected
        spec (float): percentage of non-target voxels correctly not detected
    """
    # accepts lists
    target = np.array(target)
    estimate = np.array(estimate)

    if mask is None:
        mask = np.ones(target.shape)
    else:
        # accept list
        mask = np.array(mask)

    mask = mask.astype(bool)
    target = target[mask].astype(bool)
    estimate = estimate[mask].astype(bool)

    num_true_pos = (target & estimate).sum()
    num_true = target.sum()
    sens = num_true_pos / num_true

    num_true_neg = (~target & ~estimate).sum()
    num_false = target.size - num_true
    spec = num_true_neg / num_false

    return sens, spec
