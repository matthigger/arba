import pathlib

from mh_pytools import file

from graph.report import plot_report

folder = pathlib.Path(__file__).parent

f_b0 = folder / 'b0.nii.gz'
f_part_graph = folder / 'part_graph.p.gz'
f_mask_effect = folder / 'mask_effect.nii.gz'
pg = file.load(f_part_graph)

f_out = folder / 'segment_stats.pdf'
plot_report(pg, f_out, f_mask_effect=f_mask_effect, f_back=f_b0)
