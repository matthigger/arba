import matplotlib.cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from arba.space.mask import Mask


def size_v_mu_diff(*args, **kwargs):
    def get_cov(reg):
        fs0, fs1 = reg.feat_stat.values()
        return np.linalg.norm(fs0.mu - fs1.mu)

    ylabel = r'||\mu_0 - \mu_1||_2'
    return scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_cov_det(*args, **kwargs):
    def get_cov(reg):
        fs = sum(reg.feat_stat.values())
        return np.linalg.det(fs.cov)

    ylabel = 'det cov'
    return scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_cov_tr(*args, **kwargs):
    def get_cov(reg):
        fs = sum(reg.feat_stat.values())
        return np.trace(fs.cov)

    ylabel = 'trace cov'
    return scatter_tree(*args, fnc=get_cov, ylabel=ylabel, **kwargs)


def size_v_t2(*args, **kwargs):
    ylabel = 't2 between groups'

    def get_t2(reg):
        return reg.t2

    return scatter_tree(*args, fnc=get_t2, ylabel=ylabel, **kwargs)


def size_v_wt2(*args, **kwargs):
    ylabel = 'weighted t2 between groups'

    def get_wt2(reg):
        return reg.t2 * len(reg)

    return scatter_tree(*args, fnc=get_wt2, ylabel=ylabel, **kwargs)


def size_v_error(*args, **kwargs):
    ylabel = 'MSE in t2 from vox to cluster'

    def get_error(reg):
        return reg.sq_error / len(reg)

    return scatter_tree(*args, fnc=get_error, ylabel=ylabel, **kwargs)


def size_v_pval(*args, corrected=False, size=None, **kwargs):
    if corrected:
        ylabel = 'multi compare corrected pval'

        def get_pval(reg):
            return reg.pval * size / len(reg)
    else:
        ylabel = 'pval'

        def get_pval(reg):
            return reg.pval

    return scatter_tree(*args, fnc=get_pval, ylabel=ylabel, **kwargs)


def scatter_tree(sg, fnc, ylabel, cmap=None, mask=None,
                 mask_label='% mask', reg_highlight={},
                 dict_highlight=None, edge=True, reg_list=None, ax=None,
                 log_x=True, log_y=True, txt_fnc=None):
    if cmap is None:
        cmap = matplotlib.cm.coolwarm

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
        reg_list = list(sg.nodes)
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
                try:
                    c = len([1 for ijk in reg.pc_ijk if mask[ijk]])
                except AttributeError:
                    c = reg.size
                c *= 1 / len(reg)
                node_color.append(c)

        sc = nx.draw_networkx_nodes(reg_list, pos=node_pos,
                                    nodelist=reg_list,
                                    node_color=node_color, vmin=0, vmax=1,
                                    **kwargs)

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

    # draw text over nodes
    if txt_fnc is not None:
        for reg in reg_list:
            x, y = node_pos[reg]
            plt.text(x, y, txt_fnc(reg), color='g')

    # set log scales
    if log_y:
        ax.set_yscale('log')

    if log_x:
        ax.set_xscale('log')

    # label / cleanup
    log_scale_shift = 1.2
    x_list = [x[0] for x in node_pos.values()]
    y_list = [x[1] for x in node_pos.values() if x[1] > 0]
    x_lim = min(x_list) / log_scale_shift, max(x_list) * log_scale_shift
    y_lim = min(y_list) / log_scale_shift, max(y_list) * log_scale_shift
    plt.gca().set_xbound(x_lim)
    plt.gca().set_ybound(y_lim)

    # label
    plt.xlabel('Region Size (voxels)')
    plt.ylabel(ylabel)
    plt.grid(True)

    if mask is not None:
        cb1 = plt.colorbar(sc, orientation='vertical')
        cb1.set_label(mask_label)
        # todo: kludge
        return cb1
