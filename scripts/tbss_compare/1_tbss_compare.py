import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file
from pnl_data.set import ofer_tbss
from pnl_segment.plot import plot_report

folder_out = ofer_tbss.folder / 'arba'
f_back = ofer_tbss.folder / 'arba' / 'wm_skeleton.nii.gz'


def save_fig(f_out):
    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    with PdfPages(str(f_out)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close()


sg_arba = file.load(folder_out / 'sg_arba.p.gz')
sg_arba_test = file.load(folder_out / 'sg_arba_test.p.gz')
sg_arba_test_sig = file.load(folder_out / 'sg_arba_test_sig.p.gz')

# plot size vs pval arba_test + arba_test_sig
sns.set(font_scale=1.2)
plt.scatter([len(r) for r in sg_arba],
            [r.pval for r in sg_arba],
            color='r', label='segmentation')
plt.scatter([len(r) for r in sg_arba_test],
            [r.pval for r in sg_arba_test],
            color='b', label='test')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.ylim(1e-4, 1)
plt.xlabel('size')
plt.ylabel('pval')
plt.legend()

f_size_v_pval = folder_out / f'size_vs_pval.pdf'
save_fig(f_out=f_size_v_pval)

# plot ordinal pvals
pval_seg = sorted(r.pval for r in sg_arba)
pval_test = sorted(r.pval for r in sg_arba_test)


def get_holm_thresh(n, alpha=.05):
    return [alpha / (n - idx) for idx in range(n)]


n = len(sg_arba.nodes)
n_half = int(n / 2)
thresh_full = get_holm_thresh(n)
thresh_half = get_holm_thresh(n_half)

plt.plot(pval_seg, label='segmentation')
plt.plot(pval_test, label='testing')
plt.plot(thresh_full, label=f'thresh full (n={n})', linestyle='--')
plt.plot(thresh_half, label=f'thresh half (n={n_half})', linestyle='--')

plt.gca().set_yscale('log')
plt.xlabel('ordinal pval region (different per seg + test)')
plt.ylabel('pval')
plt.legend()
f_size_v_pval = folder_out / f'pval_holm_sig.pdf'
save_fig(f_out=f_size_v_pval)

# plot report of all nominated regions (denote those which are sig)
f_report_out = folder_out / 'nominated_regions_test.pdf'
feat_x, feat_y = sorted(
    next(iter(sg_arba_test.file_tree_dict.values())).feat_list)
label_dict = {reg: 'not significant ,' for reg in sg_arba_test.nodes}
label_dict.update({reg: 'SIGNIFICANT ,' for reg in sg_arba_test_sig.nodes})
plot_report(reg_list=sg_arba_test.nodes, ft_dict=sg_arba_test.file_tree_dict,
            f_out=f_report_out, f_back=f_back, feat_x=feat_x, feat_y=feat_y,
            label_dict=label_dict)
