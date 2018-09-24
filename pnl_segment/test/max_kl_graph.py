import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from mh_pytools import file
from pnl_segment.adaptive import pval

from pnl_segment.adaptive.part_graph_factory import get_ijk_dict
from graph.region import plot_segment_seq
from graph.region import roi_and_prob_feat
from graph.scatter_tree import size_v_mahalanobis

folder = pathlib.Path(__file__).parent

f_b0 = folder / 'b0.nii.gz'
f_part_graph = folder / 'part_graph.p.gz'
f_mask_effect = folder / 'mask_effect.nii.gz'
pg = file.load(f_part_graph)


def fnc_sort(reg):
    p, r2 = pval.get_pval(reg, grp_cmp='healthy', grp_test='effect')

    return -r2, p


sns.set(font_scale=1.2)

f_out = folder / 'segment_stats.pdf'
with PdfPages(f_out) as pdf:
    size_v_mahalanobis(pg.tree_history, f_mask_track=f_mask_effect,
                       mask_label='% effect')
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    pdf.savefig(fig)
    plt.close()

    pg_span = pg.get_min_spanning_region(fnc=fnc_sort)
    size_v_mahalanobis(pg_span, f_mask_track=f_mask_effect,
                       mask_label='% effect', edge=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    pdf.savefig(fig)
    plt.close()

    # plot
    plot_segment_seq(pg_span, f_back=f_b0)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    pdf.savefig(fig)
    plt.close()

    # load original data (allows scatter)
    raw_feat = {grp: get_ijk_dict(img_iter, raw_feat=True) for grp, img_iter in
                pg.f_img_dict.items()}

    # sort by mahalanobis
    reg_history_sort = sorted(pg_span.nodes, key=fnc_sort)
    for idx, reg in enumerate(reg_history_sort[:40]):
        roi_and_prob_feat(reg, f_back=f_b0, f_mask=f_mask_effect,
                          raw_feat=raw_feat, xlim=(0, 1.3), ylim=(-.5, .5))

        fig = plt.gcf()
        fig.set_size_inches(14, 5)
        _, p = fnc_sort(reg)
        s = len(reg.pc_ijk)
        plt.suptitle(f'{idx + 1} Abnormal Region (p-value: {p:.1e}, size: {s:.0f} voxels)')
        pdf.savefig(fig)
        plt.close()
