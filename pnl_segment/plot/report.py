from collections import defaultdict

import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_roi
from tqdm import tqdm

from pnl_segment.space import PointCloud, get_ref


def plot_reg(reg, f_back, f_mask=None, ax=None):
    if f_mask is not None:
        raise NotImplementedError

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ref = get_ref(f_back)
    roi = PointCloud(reg.pc_ijk).to_mask(shape=ref.shape).astype(int)
    roi_img = nib.Nifti1Image(roi, affine=ref.affine)

    cm = plt.cm.get_cmap('viridis')
    plot_roi(roi_img=roi_img, bg_img=str(f_back), axes=ax, cmap=cm,
             draw_cross=False)


def plot_feat(reg, ft_dict, feat_x, feat_y, grp_data_dict=None, ax=None,
              verbose=False):
    # get idx of features
    ft = next(iter(ft_dict.values()))
    idx_feat_x = ft.feat_list.index(feat_x)
    idx_feat_y = ft.feat_list.index(feat_y)

    if grp_data_dict is None:
        grp_data_dict = {grp: ft.load_data(verbose=verbose) for grp, ft in
                         ft_dict.items()}

    grp_feat_data_dict = defaultdict(lambda: defaultdict(list))
    for grp, data in grp_data_dict.items():
        for ijk in reg.pc_ijk:
            x = data[ijk[0], ijk[1], ijk[2], :, :]
            grp_feat_data_dict[grp][feat_x].append(np.mean(x[idx_feat_x, :]))
            grp_feat_data_dict[grp][feat_y].append(np.mean(x[idx_feat_y, :]))

    # build new axis if need be
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # plot
    sns.set(font_scale=1.2)
    grp_list = sorted(grp_data_dict.keys())
    cm = plt.cm.get_cmap('Set1')
    grp_color_dict = {grp: cm(idx) for idx, grp in enumerate(grp_list)}
    for grp, feat_data_dict in grp_feat_data_dict.items():
        x = feat_data_dict[feat_x]
        y = feat_data_dict[feat_y]
        if len(x) > 1000:
            x = np.random.choice(x, size=1000, replace=False)
            y = np.random.choice(y, size=1000, replace=False)
        ax.scatter(x, y, color=grp_color_dict[grp], label=grp, alpha=.5)

    for grp, feat_data_dict in grp_feat_data_dict.items():
        x = feat_data_dict[feat_x]
        y = feat_data_dict[feat_y]
        ax.scatter(np.mean(x), np.mean(y), marker='s', alpha=1,
                   color=grp_color_dict[grp], edgecolors='k')

    plt.legend()
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)


# https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
def ordinal(n):
    return "%d%s" % (
        n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


def plot_report(reg_list, ft_dict, f_out, feat_x, feat_y, f_mask=None,
                f_back=None, verbose=True, label_dict=None):
    # load data
    grp_data_dict = {grp: ft.load_data(verbose=verbose)
                     for grp, ft in ft_dict.items()}

    tqdm_dict = {'disable': not verbose,
                 'desc': 'print graph per region',
                 'total': len(reg_list)}

    sns.set(font_scale=1.2)
    with PdfPages(f_out) as pdf:
        # sort by pval
        reg_list = sorted(reg_list, key=lambda r: r.pval)
        for idx, reg in tqdm(enumerate(reg_list), **tqdm_dict):
            fig = plt.figure()

            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

            ax = plt.subplot(gs[0]), plt.subplot(gs[1])

            plot_feat(reg, ft_dict, ax=ax[1], feat_x=feat_x, feat_y=feat_y,
                      grp_data_dict=grp_data_dict, verbose=verbose)
            plot_reg(reg, f_back, f_mask, ax=ax[0])

            fig.set_size_inches(14, 5)
            if label_dict is None:
                label = ''
            else:
                label = label_dict[reg]
            label += f'{ordinal(idx + 1)} Abnormal Region: ' + \
                     f'pval: {reg.pval:.3e}, ' + \
                     f'size: {len(reg):.0f} vox'
            plt.suptitle(label)

            pdf.savefig(fig)
            plt.close()
