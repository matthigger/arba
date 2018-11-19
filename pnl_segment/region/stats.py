import numpy as np


def BenHochYek(p_list, sig=.05):
    """ returns threshold to reject null

    assumes: hypotheses are positively correlated

    https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg%E2%80%93Yekutieli_procedure
    Args:
        p_list (list): p values

    Returns:
        thresh (float): threshold to reject @

    >>> import numpy as np
    >>> p_list = np.logspace(0, -3, 10)
    >>> BenHochYek(p_list)
    0.030000000000000006
    >>> p_list = np.logspace(0, -3, 100)
    >>> BenHochYek(p_list)
    0.0235
    """

    raise NotImplementedError('needs testing')

    c_m = 1
    m = len(p_list)

    # sort in descending
    p_list = sorted(p_list, reverse=True)

    for k, p in enumerate(p_list):
        # count backward / 'correct' python idx
        k = m - k + 1

        # compute thresh
        thresh = k * sig / (c_m * m)

        if p < thresh:
            return thresh

    return 0


def HolmBonf(p_list, sig=.05):
    """
    >>> p_list = [.01, .04, .005, .03]
    >>> [p for p in p_list if p < HolmBonf(p_list)]
    [0.01, 0.005]
    >>> p_list = [.01, .02]
    >>> [p for p in p_list if p < HolmBonf(p_list)]
    [0.01, 0.02]
    >>> p_list = [.05, .05]
    >>> [p for p in p_list if p < HolmBonf(p_list)]
    []
    """
    p_list = sorted(p_list)
    m = len(p_list)

    for k, p in enumerate(p_list):
        if p > sig / (m - k):
            break
        k += 1
    if k == m:
        # all are significant
        return 1
    elif k == 0:
        # nothing is significant
        return 0
    else:
        return np.mean((p_list[k], p_list[k - 1]))
