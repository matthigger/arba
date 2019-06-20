import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn.plotting import plot_roi

import arba.bayes
from .norm2d import plot_norm_2d
from .save_fig import save_fig

sns.set(font_scale=1.2)


def plot_delta(mask, reg, f_bg=None, feat_list=None, f_out=None,
               feat_xylim=None, mask_target=None, delta_xylim=None, **kwargs):
    # perform bayes estimate of mu and cov
    grp_mu_dict, _ = reg.bayes()

    fig = plt.figure()
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 1))

    # plot mask of position
    f_roi = mask.to_nii()
    display = plot_roi(roi_img=str(f_roi), bg_img=str(f_bg),
                       display_mode='ortho', figure=fig, axes=ax0)

    if mask_target is not None:
        f_mask_target = str(mask_target.to_nii())
        display.add_contours(img=f_mask_target, colors='r', levels=[.5])

    # plot distribution of features per grp
    patch_list = list()
    plt.sca(ax1)
    for (grp, (mu, cov)), color in zip(grp_mu_dict.items(), 'bg'):
        _, p = plot_norm_2d(mu, cov, plot_mu=True, label=grp, ax=ax1,
                            facecolor=color, alpha=.5)
        patch_list.append(p)
    plt.legend(handles=patch_list)

    # plot distribution of difference between grp features
    (grp0, grp1), delta_mu, delta_cov = \
        arba.bayes.bayes_mu_delta(grp_mu_dict=grp_mu_dict, **kwargs)
    label = f'{grp1} - {grp0}'
    _, p = plot_norm_2d(delta_mu, delta_cov, plot_mu=True, label=label,
                        ax=ax2, facecolor='r', alpha=.5)
    plt.sca(ax2)
    plt.legend(handles=[p])

    # label axes of feature plots
    if feat_list is not None:
        for _ax in [ax1, ax2]:
            plt.sca(_ax)
            plt.xlabel(feat_list[0])
            plt.ylabel(feat_list[1])

    # set limits
    set_lim(ax1, mu_cov_iter=grp_mu_dict.values())
    set_lim(ax2, mu_cov_iter=[(delta_mu, delta_cov)])

    # set limits (manually)
    for ax, xylim in ((ax1, feat_xylim),
                      (ax2, delta_xylim)):
        if xylim is None:
            continue
        ax.set_xlim(xylim[0])
        ax.set_ylim(xylim[1])

    if f_out is not None:
        save_fig(f_out=f_out, size_inches=(10, 10))

    return ax0, ax1, ax2


def set_lim(ax, **kwargs):
    min_x, max_x = get_lim(**kwargs)

    ax.set_xlim((min_x[0], max_x[0]))
    ax.set_ylim((min_x[1], max_x[1]))


def get_lim(mu_cov_iter, n_std=4):
    min_x = np.ones(2) * np.inf
    max_x = -min_x

    for (mu, cov) in mu_cov_iter:
        std = np.diag(cov) ** .5
        min_x = np.minimum(min_x, mu - n_std * std)
        max_x = np.maximum(max_x, mu + n_std * std)

    return min_x, max_x
