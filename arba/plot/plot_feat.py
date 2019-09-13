import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arba.space import Mask, PointCloud

sns.set(font_scale=1.2)


def plot_feat(data_img, sbj_bool, feat_tuple, mask=None, pc=None, ijk=None,
              ax=None, max_pts=1000, **kwargs):
    """ makes a scatter of features in some region

    Args:
        data_img (DataImage): data set to plot
        sbj_bool (tuple): sbj to plot
        feat_tuple (tuple): features to plot
        ax (plt.Axes): axes to plot on
        mask (Mask):    defines space
        pc (PointCloud):defines space
        ijk (tuple):    defines space
        max_pts (int): maximum number of pts to plot (sample if need be)

    Returns:
        ax (plt.Axes)
    """

    assert all(feat in data_img.feat_list for feat in feat_tuple), \
        'invalid feat'

    assert 1 == sum(((mask is not None),
                     (pc is not None),
                     (ijk is not None))), 'either mask xor pc xor ijk required'

    # get mask
    if pc is not None:
        mask = pc.to_mask()
    elif ijk is not None:
        mask = np.zeros(data_img.ref.shape)
        i, j, k = ijk
        mask[i, j, k] = 1

    # get ax
    if ax is None:
        _, ax = plt.subplots(1, 1)

    with data_img.loaded():
        # get relevant data
        _data = data_img.data[mask, :, :]
        _data = _data[sbj_bool, :]

        x = _data[:, data_img.feat_list.index(feat_tuple[0])]
        y = _data[:, data_img.feat_list.index(feat_tuple[1])]

        # sub-sample if needed
        if x.size > max_pts:
            idx = np.random.choice(range(x.size), size=max_pts, replace=False)
            x = x[idx]
            y = y[idx]

        ax.scatter(x, y, **kwargs)
        ax.set_xlabel(feat_tuple[0])
        ax.set_ylabel(feat_tuple[1])

    return ax
