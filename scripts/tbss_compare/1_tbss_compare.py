from mh_pytools import file
from pnl_data.set import ofer_tbss
from pnl_segment.plot import plot_report

folder_out = ofer_tbss.folder / 'arba'
f_back = ofer_tbss.folder / 'arba' / 'wm_skeleton.nii.gz'
f_report_out = folder_out / 'atypical_reg.pdf'

# load
ft_dict_test = file.load(folder_out / 'ft_dict_test.p.gz')
sg_hist = file.load(folder_out / 'sg_hist_reduced.p.gz')

# plot report of sig regions
sg = file.load(folder_out, 'sg_arba_test_sig.p.gz')
feat_x, feat_y = sorted(next(iter(sg.file_tree_dict.values())).feat_list)
plot_report(reg_list=sg.nodes, ft_dict=ft_dict_test, f_out=f_report_out,
            f_back=f_back, feat_x=feat_x, feat_y=feat_y)
