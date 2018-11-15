import matplotlib as mpl
import networkx as nx
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pnl_segment.space.mask import Mask
from pnl_segment.region import RegionMaha


def size_v_mu_diff(*args, **kwargs):
    def get_cov(reg):
        fs0, fs1 = reg.feat_stat.values()
        return np.linalg.norm(fs0.mu - fs1.mu)

    ylabel = '||\mu_0 - \mu_1||_2'
    scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_cov_det(*args, **kwargs):
    def get_cov(reg):
        fs = sum(reg.feat_stat.values())
        return np.linalg.det(fs.cov)

    ylabel = 'det cov'
    scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_cov_tr(*args, **kwargs):
    def get_cov(reg):
        fs = sum(reg.feat_stat.values())
        return np.trace(fs.cov)

    ylabel = 'trace cov'
    scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_mahalanobis(*args, **kwargs):
    ylabel = 'mahalanobis to healthy'

    def get_maha(reg):
        return RegionMaha.get_obj(reg) * len(reg)

    scatter_tree(*args, fnc=get_maha, ylabel=ylabel, **kwargs)


def scatter_tree(sg, fnc, ylabel, cmap=mpl.cm.coolwarm, mask=None,
                 mask_label='% mask', reg_highlight=[],
                 dict_highlight=None, edge=True, reg_list=None, ax=None,
                 log_x=True, log_y=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    sns.set()
    if dict_highlight is None:
        dict_highlight = {'linewidths': 3, 'edgecolors': 'k'}

    # get pts with effect mask
    if mask is not None and not isinstance(mask, Mask):
        mask = Mask.from_nii(mask)

    # get node_pos
    node_pos = dict()
    if reg_list is None:
        reg_list = sg.nodes
    reg_list = set(reg_list) | set(reg_highlight)

    for reg in reg_list:
        size = len(reg)

        node_pos[reg] = size, fnc(reg)

    def plot_node(reg_list, **kwargs):

        if mask is None:
            node_color = None
        else:
            node_color = list()
            for reg in reg_list:
                # get color
                c = len([1 for ijk in reg.pc_ijk if mask[ijk]])
                c *= 1 / len(reg)
                node_color.append(c)

        sc = nx.draw_networkx_nodes(reg_list, pos=node_pos,
                                    nodelist=reg_list,
                                    node_color=node_color, **kwargs)

        return sc

    # draw non highlight nodes
    sc = plot_node(set(reg_list) - set(reg_highlight), cmap=cmap)

    # draw highlight nodes
    if reg_highlight:
        plot_node(reg_highlight, cmap=cmap, **dict_highlight)

    # draw edges
    if edge:
        reg_set = set(reg_list)
        edgelist = [e for e in sg.edges if set(e).issubset(reg_set)]
        nx.draw_networkx_edges(sg, pos=node_pos, edgelist=edgelist)

    # label / cleanup
    if log_y:
        ax.set_yscale('log')

    if log_x:
        ax.set_xscale('log')
    log_scale_shift = 1.2
    x_list = [x[0] for x in node_pos.values()]
    y_list = [x[1] for x in node_pos.values()]
    x_lim = min(x_list) / log_scale_shift, max(x_list) * log_scale_shift
    y_lim = np.percentile(y_list, 5) / log_scale_shift, max(
        y_list) * log_scale_shift
    plt.gca().set_xbound(x_lim)
    plt.gca().set_ybound(y_lim)
    plt.xlabel('size (vox)')
    plt.ylabel(ylabel)
    if mask is not None:
        cb1 = plt.colorbar(sc, orientation='vertical')
        cb1.set_label(mask_label)
    plt.minorticks_on()
