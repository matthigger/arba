import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from pnl_segment import plot
from pnl_segment.seg_graph import FeatStat, seg_graph_factory
from pnl_segment.simulate import Model


def build_and_reduce(model, num_sbj_per_grp, folder):
    # build file_tree by generating nifti sampled from model
    ft_dict = dict()
    for idx in range(2):
        label = f'grp_{idx}'
        _folder = pathlib.Path(folder) / label
        os.makedirs(_folder, exist_ok=True)
        ft_dict[label] = model.to_file_tree(n=num_sbj_per_grp,
                                            folder=_folder)

    # build and reduce a seg_graph
    sg_hist = seg_graph_factory(obj='maha', file_tree_dict=ft_dict,
                                history=True)
    sg_hist.reduce_to(1, verbose=verbose)
    return sg_hist


def print_graph(sg_hist, folder):
    sns.set(font_scale=1.2)
    with PdfPages(str(folder / 'hist_cdf.pdf')) as pdf:
        plt.hist([r.pval for r in sg_hist.tree_history],
                 bins=np.logspace(-3, 0, 100))
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

    with PdfPages(str(folder / 'size_vs_error.pdf')) as pdf:
        plot.size_vs_error_normed(sg_hist=sg_hist)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

    with PdfPages(str(folder / 'size_vs_maha.pdf')) as pdf:
        plot.size_v_mahalanobis(sg=sg_hist.tree_history,
                                dict_highlight={'linewidths': 3,
                                                'edgecolors': 'k'},
                                log_x=True,
                                log_y=True)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

    with PdfPages(str(folder / 'size_vs_wmaha.pdf')) as pdf:
        plot.size_v_wmahalanobis(sg=sg_hist.tree_history,
                                 dict_highlight={'linewidths': 3,
                                                 'edgecolors': 'k'},
                                 log_x=True,
                                 log_y=True)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

    with PdfPages(str(folder / 'size_vs_pval.pdf')) as pdf:
        ax = plt.gca()
        plot.size_v_pval(sg=sg_hist.tree_history,
                         dict_highlight={'linewidths': 3,
                                         'edgecolors': 'k'},
                         log_x=True,
                         log_y=False,
                         ax=ax)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    from mh_pytools import parallel

    par_flag = True

    n_iter = 4
    num_sbj_per_grp = 10
    shape = (10, 10, 10)
    verbose = True
    folder_out = pathlib.Path('/home/matt/Downloads/big_cov_10')
    os.makedirs(folder_out, exist_ok=True)
    mode = 'gradient_mu'

    # build model of standard normal per vox iid
    ijk_fs_dict = dict()
    if mode == 'gradient_mu':
        for ijk in np.ndindex(shape):
            ijk_fs_dict[ijk] = FeatStat(n=10000, mu=ijk[0], cov=100)
    if mode == 'gradient_cov':
        for ijk in np.ndindex(shape):
            ijk_fs_dict[ijk] = FeatStat(n=10000, mu=0, cov=ijk[0] + 1)
    elif mode == 'constant':
        for ijk in np.ndindex(shape):
            ijk_fs_dict[ijk] = FeatStat(n=10000, mu=0, cov=1)
    m = Model(ijk_fs_dict, shape=shape)


    def fnc(idx):
        np.random.seed(idx)
        folder = folder_out / f'pval_cook_{idx}'
        sg_hist = build_and_reduce(model=m, num_sbj_per_grp=num_sbj_per_grp,
                                   folder=folder)
        print_graph(sg_hist, folder=folder)


    if par_flag:
        arg_list = [{'idx': idx} for idx in range(n_iter)]
        parallel.run_par_fnc(fnc, arg_list=arg_list)
    else:
        for idx in range(n_iter):
            fnc(idx)
