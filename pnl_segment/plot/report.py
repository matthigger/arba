import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pnl_segment.adaptive import pval
from pnl_segment.adaptive.part_graph_factory import get_ijk_dict

from graph.region import plot_stat, roi_and_prob_feat
from graph.scatter_tree import size_v_mahalanobis


def fnc_sort(reg, grp_cmp='healthy', grp_test='effect'):
    p, r2 = pval.get_pval(reg, grp_cmp=grp_cmp, grp_test=grp_test)

    return -r2, p


def plot_report(pg, f_out, f_mask_effect=None, f_back=None):
    sns.set(font_scale=1.2)
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
        reg_dict = {reg: -np.log10(fnc_sort(reg)[1]) for reg in pg_span}
        plot_stat(reg_dict, f_back=f_back, label='-log10 pval')
        fig = plt.gcf()
        fig.set_size_inches(14, 5)
        pdf.savefig(fig)
        plt.close()

        # load original data (allows scatter)
        raw_feat = {grp: get_ijk_dict(img_iter, raw_feat=True) for
                    grp, img_iter in
                    pg.f_img_dict.items()}

        # sort by mahalanobis
        reg_history_sort = sorted(pg_span.nodes, key=fnc_sort)
        for idx, reg in enumerate(reg_history_sort[:40]):
            roi_and_prob_feat(reg, f_back=f_back, f_mask=f_mask_effect,
                              raw_feat=raw_feat, xlim=(0, 1.3), ylim=(-.5, .5))

            fig = plt.gcf()
            fig.set_size_inches(14, 5)
            neg_r2, p = fnc_sort(reg)
            s = len(reg.pc_ijk)
            label = f'{idx + 1} Abnormal Region ' + \
                    f'(p-value: {p:.1e}, ' \
                    f'mah_dist: {-neg_r2:.1f}, ' + \
                    f'size: {s:.0f} voxels)'
            plt.suptitle(label)
            pdf.savefig(fig)
            plt.close()
