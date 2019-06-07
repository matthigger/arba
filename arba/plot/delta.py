import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import plot_roi

from .norm2d import plot_norm_2d
from .save_fig import save_fig

sns.set(font_scale=1.2)


def plot_delta(mask, grp_mu_cov_dict, f_bg=None, feat_list=None, f_out=None,
               feat_xylim=None, delta_xylim=None, mask_target=None):
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
    for (grp, (mu, cov)), color in zip(grp_mu_cov_dict.items(), 'bg'):
        _, p = plot_norm_2d(mu, cov, plot_mu=True, label=grp, ax=ax1,
                            facecolor=color, alpha=.5)
        patch_list.append(p)
    plt.legend(handles=patch_list)

    # plot distribution of difference between grp features
    grp0, grp1 = sorted(grp_mu_cov_dict.keys())
    mu0, cov0 = grp_mu_cov_dict[grp0]
    mu1, cov1 = grp_mu_cov_dict[grp1]
    delta_mu = mu1 - mu0
    delta_cov = cov0 + cov1
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
    for ax, xylim in ((ax1, feat_xylim),
                      (ax2, delta_xylim)):
        if xylim is None:
            continue
        ax.set_xlim(xylim[0])
        ax.set_ylim(xylim[1])

    if f_out is not None:
        save_fig(f_out=f_out, size_inches=(10, 10))
