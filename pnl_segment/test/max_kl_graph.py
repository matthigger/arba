import pathlib

from mh_pytools import file

from graph.report import plot_report

folder = pathlib.Path(__file__).parent
folder_data = folder / 'data'
f_b0 = folder_data / 'b0.nii.gz'
f_part_graph = folder_data / 'seg_graph.p.gz'
f_mask_effect = folder_data / 'mask_effect.nii.gz'
pg = file.load(f_part_graph)

f_out = folder / 'pdf' / 'segment_stats.pdf'
plot_report(pg, f_out, f_mask_effect=f_mask_effect, f_back=f_b0)
