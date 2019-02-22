import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import pnl_data
from mh_pytools import file
from arba.plot.scatter_tree import size_v_mahalanobis, size_v_pval
from arba.space import Mask

# param
alpha = .05
folder = pnl_data.folder_data / 'arba_toy_ex'

# load
sg_hist = file.load(folder / 'save' / 'sg_hist_seg.p.gz')
_, tree_hist_resolved = sg_hist.resolve_hist()
sg_arba = sg_hist.cut_greedy_sig(alpha=alpha)
region_spaces = list(reg.pc_ijk for reg in sg_arba.nodes)
reg_highlight = [reg for reg in tree_hist_resolved.nodes
                 if reg.pc_ijk in region_spaces]
mask = Mask.from_nii(folder / 'mask_effect.nii.gz')

# print graphs
sns.set(font_scale=1)


def print_fig(f_out):
    with PdfPages(f_out) as pdf:
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()


f_out = folder / 'toy_size_v_maha.pdf'
cb = size_v_mahalanobis(tree_hist_resolved, mask=mask,
                        reg_highlight=reg_highlight)
plt.ylabel('Maha(r)')
cb.remove()
print_fig(f_out)

f_out = folder / 'toy_size_v_pval.pdf'
size_v_pval(tree_hist_resolved, mask=mask, reg_highlight=reg_highlight,
            mask_label='% region with effect')
plt.ylabel('\'p-value\'')
print_fig(f_out)
