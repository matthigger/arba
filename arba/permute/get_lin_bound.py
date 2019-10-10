import numpy as np


def get_lin_bound(x, y):
    """ gets a line (fnc x -> y) which is greater than each y

    Args:
        x (np.array): (num_pts) domain
        y (np.array): (num_pts) range

    Returns:
        fnc (fnc): linear function which maps x to y
    """
    # get point (of max y value)
    max_idx = np.argmax(y)
    max_x = x[max_idx]
    max_y = y[max_idx]

    # get slope to contain all other points
    _x = x - max_x
    _y = y - max_y
    with np.seterr(divide='ignore'):
        _m = _y / _x
    _m = _m[~np.isnan(_m)]
    m_idx = np.argmin(np.abs(_m))
    m = _m[m_idx]

    return lambda x: (x - max_x) * m + max_y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = 10
    x = np.random.standard_normal(n)
    y = np.random.standard_normal(n)

    fnc = get_lin_bound(x, y)
    y_bnd = fnc(x)

    sns.set()
    plt.scatter(x, y)
    plt.plot(x, y_bnd)
    plt.show()
