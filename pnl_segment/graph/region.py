import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from nilearn import plotting
from scipy.stats import multivariate_normal


def get_meshgrid(mv_norm, idx_to_plot, std_dev_zoom=3, n=100):
    """ gets range of values for zooming to std_dev_zoom left / right of mu

    >>> mean = np.array([0, 0])
    >>> cov = np.eye(2) @ np.array([1, 2]) ** 2
    >>> mv_norm = multivariate_normal(mean, cov)
    >>> mv_norm.cov
    array([[1., 0.],
           [0., 4.]])
    >>> get_norm_range(mv_norm)
    array([[-3.,  3.],
           [-6.,  6.]])
    """
    std_dev = mv_norm.cov[idx_to_plot, idx_to_plot] ** .5
    mu = mv_norm.mean[idx_to_plot]
    lim = np.vstack((mu - std_dev * std_dev_zoom,
                     mu + std_dev * std_dev_zoom)).T
    ax_val = [np.linspace(btm, top, n) for btm, top in lim]
    x, y = np.meshgrid(*ax_val)
    domain = np.vstack((x.flatten('F'), y.flatten('F'))).T
    return x, y, domain


def prob_feat(reg, grp_list=None, idx_to_plot=[0, 1], idx_label=None,
              n_level=4, cmap=cm.Set1, ax=None, **kwargs):
    if grp_list is None:
        grp_list = reg.feat_stat.keys()

    if idx_label is None:
        fs = next(iter(reg.feat_stat.values()))
        if fs.label is None:
            idx_label = tuple(f'idx {idx}' for idx in idx_to_plot)
        else:
            idx_label = fs.label[idx_to_plot[0]], \
                        fs.label[idx_to_plot[1]]

    if ax is None:
        _, ax = plt.subplots()
    plt.sca(ax)

    xyp_dict = dict()
    for grp in grp_list:
        # build multivariate_normal
        mv_norm = reg.feat_stat[grp].to_normal(**kwargs)

        # find range to plot
        x, y, domain = get_meshgrid(mv_norm, idx_to_plot=idx_to_plot)

        # find prob
        p = mv_norm.pdf(domain).reshape(len(y), len(x))

        # store (allows computing levels all-together for consistency)
        xyp_dict[grp] = x, y, p

    # # compute levels
    # p_all = np.hstack(p for _, _, p in xyp_dict.values())
    # levels = sorted(np.percentile(p_all, 100 - np.geomspace(1, 40, n_level)))

    # plot
    legend_dict = dict()
    for idx, (grp, (x, y, p)) in enumerate(xyp_dict.items()):
        color = rgb2hex(cmap(idx))
        plt.contour(x, y, p, n_level, label=grp, colors=color)
        plt.xlabel(idx_label[0])
        plt.ylabel(idx_label[1])
        legend_dict[grp] = Line2D(x[0], y[0], color=color, lw=2)

    plt.legend(legend_dict.values(), legend_dict.keys())


def roi_and_prob_feat(reg, f_back=None, **kwargs):
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    prob_feat(reg, ax=ax[1], **kwargs)

    # build mask of reg
    _, f_reg_mask = tempfile.mkstemp(suffix='.nii.gz')
    reg.pc_ijk.to_nii(f_reg_mask)

    # plot
    disp = plotting.plot_roi(str(f_reg_mask), bg_img=str(f_back), axes=ax[0],
                             draw_cross=False, dim=-1)
    disp.add_contours(str(f_reg_mask), filled=True, levels=[0.5], colors='g')
