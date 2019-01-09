import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file
from pnl_data.set import ofer_tbss
from pnl_segment.plot import plot_report, size_v_pval
import seaborn as sns

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

sns.set(font_scale=1.2)
plt.scatter([len(r) for r in sg_arba],
            [r.pval for r in sg_arba],
            color='r', label='segmentation')
plt.scatter([len(r) for r in sg_arba_test],
            [r.pval for r in sg_arba_test],
            color='b', label='test')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')

# plot report of sig regions
f_report_out = folder_out / 'atypical_reg.pdf'
sg = file.load(folder_out / 'sg_arba_test_sig.p.gz')
feat_x, feat_y = sorted(next(iter(sg.file_tree_dict.values())).feat_list)
plot_report(reg_list=sg.nodes, ft_dict=sg.file_tree_dict, f_out=f_report_out,
            f_back=f_back, feat_x=feat_x, feat_y=feat_y)

# plot pval vs size
f_size_v_pval = folder_out / f'size_vs_pval.pdf'
sg_hist = file.load(folder_out / 'sg_hist.p.gz')
size_v_pval(sg=sg_hist.tree_history,
            log_x=True,
            log_y=True)
save_fig(f_out=f_size_v_pval)
