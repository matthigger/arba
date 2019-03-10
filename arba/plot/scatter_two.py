import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from tqdm import tqdm

from arba.plot.save_fig import save_fig

sns.set(font_scale=1.2)


def det_cov(reg):
    fs0, fs1 = reg.fs_dict.values()
    cov = (fs0.n * fs0.cov +
           fs1.n * fs1.cov) / (fs0.n + fs1.n)
    return np.linalg.det(cov)


def wt_squared(reg):
    return reg.t2 * len(reg)


def pval(reg):
    return -np.log10(reg.pval)


def sq_error(reg):
    return reg.sq_error[0] / len(reg)


def obj(reg):
    return len(reg) * reg.t2 / max(sq_error(reg), 1) ** 2


def scatter_tree_two(g, fnc_x, fnc_y, xlabel=None, ylabel=None, logx=False,
                     logy=False, nodelist=None, fnc_color=None, verbose=False,
                     f_out=None, fnc_size=None):
    # restrict to nodes / edges
    if nodelist is None:
        nodelist = list(g.nodes)
    else:
        nodelist = list(nodelist)
    nodeset = set(nodelist)
    edgelist = [(n0, n1) for (n0, n1) in g.edges
                if set((n0, n1)).issubset(nodeset)]

    # get labels of each axis
    if xlabel is None:
        xlabel = fnc_x.__name__

    if ylabel is None:
        ylabel = fnc_y.__name__

    # compute node positions
    node_pos = dict()
    tqdm_dict = {'disable': not verbose,
                 'desc': 'computing node positions'}
    for n in tqdm(nodelist, **tqdm_dict):
        node_pos[n] = fnc_x(n), fnc_y(n)

    # compute node color
    if fnc_color is None:
        node_color = 'r'
        vmin = 0
        vmax = 1
        cmap = None
    else:
        node_color = list()
        tqdm_dict['desc'] = 'compute node color'
        for r in tqdm(nodelist, **tqdm_dict):
            node_color.append(fnc_color(r))
        vmin = min(node_color)
        vmax = max(node_color)
        cmap = plt.get_cmap('inferno')

    # compute node color
    if fnc_size is None:
        node_size = 300
    else:
        node_size = list()
        tqdm_dict['desc'] = 'compute node size'
        for r in tqdm(nodelist, **tqdm_dict):
            node_size.append(fnc_size(r))

        node_size = np.array(node_size)
        node_size -= np.mean(node_size)
        node_size *= 75 / np.std(node_size)
        node_size += 300

    # prep figure + axis
    fig, ax = plt.subplots(1, 1)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    # plot
    nx.draw_networkx_edges(g, pos=node_pos, edgelist=edgelist)
    nx.draw_networkx_nodes(g, pos=node_pos, nodelist=nodelist,
                           node_color=node_color, vmin=vmin, vmax=vmax,
                           cmap=cmap, node_size=node_size)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return save_fig(f_out)


if __name__ == '__main__':
    from pnl_data.set.hcp_100 import folder
    from mh_pytools import file
    import random

    num_leafs = 5
    folder = folder / 'arba_cv_MF_FA-MD' / 'save'
    f_tree_hist_reg = folder / 'tree_hist_reg.p.gz'
    fnc_x = len
    fnc_y = obj
    fnc_color = pval
    f_out = folder.parent / f'{fnc_x.__name__}_{fnc_y.__name__}_{fnc_color.__name__}.pdf'

    if f_tree_hist_reg.exists():
        tree_hist_reg = file.load(f_tree_hist_reg)
        print('done load')
    else:
        # computationally expensive, save for reruns
        sg_hist_seg = file.load(folder / 'sg_hist_seg.p.gz')
        node_reg_dict, tree_hist_reg = sg_hist_seg.resolve_hist(verbose=True)
        file.save(tree_hist_reg, f_tree_hist_reg)
        print('done save')

    # subsample num_leafs
    node_set = set()
    leaf_set = [r for r in tree_hist_reg.nodes if len(r) == 1]
    random.seed(1)
    for leaf in random.choices(leaf_set, k=num_leafs):
        node_set.add(leaf)
        node_set |= nx.descendants(tree_hist_reg, leaf)

    # plot
    scatter_tree_two(tree_hist_reg,
                     fnc_x=fnc_x, fnc_y=fnc_y, fnc_color=fnc_color,
                     logx=True, logy=True, nodelist=node_set,
                     verbose=True, f_out=f_out)
