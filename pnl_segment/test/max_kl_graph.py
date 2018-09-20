import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from mh_pytools import file
from pnl_segment.adaptive import pval

from graph.scatter_tree import size_v_mahalanobis

folder = pathlib.Path(__file__).parent

f_part_graph = folder / 'part_graph.p.gz'
f_mask_effect = folder / 'mask_effect.nii.gz'
pg = file.load(f_part_graph)


def fnc_sort(reg):
    p, r2 = pval.get_pval(reg, grp_cmp='healthy', grp_test='effect')

    return -r2


f_out = folder / 'size_v_mahalanobis.pdf'
with PdfPages(f_out) as pdf:
    sns.set(font_scale=1.2)
    pg_span = pg.get_min_spanning_region(fnc=fnc_sort)
    size_v_mahalanobis(pg, f_mask_track=f_mask_effect,
                       mask_label='% effect',
                       reg_highlight=pg_span.nodes)
    plt.show()
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    pdf.savefig(fig)
