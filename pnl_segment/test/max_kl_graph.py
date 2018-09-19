import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import seaborn as sns
from mh_pytools import file
from pnl_segment.adaptive import pval

from matplotlib.backends.backend_pdf import PdfPages

folder = pathlib.Path(__file__).parent

f_part_graph = folder / 'part_graph.p.gz'
f_mask_effect = folder / 'mask_effect.nii.gz'
pg = file.load(f_part_graph)


def graph_size_v_mahalanobis(pg, cmap=mpl.cm.coolwarm, f_mask_track=None,
                             mask_label='% mask', reg_highlight=[],
                             dict_highlight=None):
    if dict_highlight is None:
        dict_highlight = {'linewidths': 3, 'edgecolors': 'k'}

    # get pts with effect mask
    if f_mask_track is not None:
        mask = nib.load(str(f_mask_track)).get_data()
        mask_ijk = set(tuple(x) for x in np.vstack(np.where(mask)).T)

    # get node_pos
    node_pos = dict()
    for reg in pg.tree_history:
        size = len(reg)
        p, r2 = pval.get_pval(reg, grp_cmp='healthy', grp_test='effect')

        node_pos[reg] = size, r2

    def plot_node(reg_list, **kwargs):

        if f_mask_track is None:
            node_color = None
        else:
            node_color = list()
            for reg in reg_list:
                # get color
                c = len([1 for ijk in reg.pc_ijk.x if tuple(ijk) in mask_ijk])
                c *= 1 / len(reg)
                node_color.append(c)

        sc = nx.draw_networkx_nodes(pg.tree_history, pos=node_pos,
                                    nodelist=reg_list,
                                    node_color=node_color, **kwargs)

        return sc

    # draw non highlight nodes
    reg_list = set(pg.tree_history.nodes) - set(reg_highlight)
    sc = plot_node(reg_list, cmap=cmap)

    # draw highlight nodes
    if reg_highlight:
        plot_node(reg_highlight, cmap=cmap, **dict_highlight)

    # draw edges
    nx.draw_networkx_edges(pg.tree_history, pos=node_pos)

    # label / cleanup
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_xbound(.8, max(x[0] for x in node_pos.values()))
    plt.xlabel('size (vox)')
    plt.ylabel('mahalanobis to healthy')
    if f_mask_track is not None:
        cb1 = plt.colorbar(sc, orientation='vertical')
        cb1.set_label(mask_label)


def fnc_sort(reg):
    p, r2 = pval.get_pval(reg, grp_cmp='healthy', grp_test='effect')

    return -r2


f_out = folder / 'size_v_mahalanobis.pdf'
with PdfPages(f_out) as pdf:
    sns.set(font_scale=1.2)
    pg_span = pg.get_min_spanning_region(fnc=fnc_sort)
    graph_size_v_mahalanobis(pg, f_mask_track=f_mask_effect,
                             mask_label='% effect',
                             reg_highlight=pg_span.nodes)
    plt.show()
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    pdf.savefig(fig)
