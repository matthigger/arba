import pymc3 as pm

import numpy as np
from arba.region import FeatStat


def get_data(file_tree, mask, split):
    """ gets data associated with file tree, mask and split

    Args:
        file_tree (FileTree): data object
        mask (Mask): location in image to observe
        split (Split): split object (determiens which sbj are in which group)
    """
    assert file_tree.is_loaded, 'file tree must be loaded'
    assert split.sbj_list == file_tree.sbj_list, 'sbj_list mismatch'
    assert len(split) == 2, 'split must be of len 2'

    x = file_tree.data[mask, ...]

    data = list()
    for grp, grp_bool in split.bool_iter():
        _x = x[:, grp_bool, :]
        _x = np.reshape(_x, newshape=(-1, _x.shape[-1]), order='F')
        data.append(_x)

    return data[0], data[1]


def estimate_delta(data0, data1, **kwargs):
    """ estimates delta between observations

    Args:
        data0 (np.array): (num_samples0, d) observations grp 0
        data1 (np.array): (num_samples1, d) observations grp 1

    Returns:
        model (pm.Model)
        trace
    """
    # todo: use rigorous 'non-informative' prior
    fs = sum(FeatStat.from_array(x.T) for x in (data0, data1))

    model = pm.Model()

    d = data0.shape[1]

    with model:
        # todo: how to set eta and std of HalfCauchy?
        packed_L = pm.LKJCholeskyCov('packed_L', n=d,
                                     eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
        L = pm.expand_packed_triangular(d, packed_L)

        mu_0 = pm.MvNormal('mu_0', fs.mu, fs.cov, shape=d,
                           testval=data0.mean(axis=0))
        mu_1 = pm.MvNormal('mu_1', fs.mu, fs.cov, shape=d,
                           testval=data1.mean(axis=0))

        obs_1 = pm.MvNormal('obs_1', mu_1, chol=L, observed=data1)
        obs_0 = pm.MvNormal('obs_0', mu_0, chol=L, observed=data0)

        delta = pm.Deterministic('delta', mu_1 - mu_0)

        trace = pm.sample(**kwargs)

    return model, trace
