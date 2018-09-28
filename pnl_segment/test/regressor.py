import pathlib
from collections import defaultdict
from random import choice

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import pnl_data
from mh_pytools import file
from pnl_data.set.intrust import get_sbj
from pnl_segment.vox_regress.polyregressor import PolyRegressor

n_sbj = 1e9
x_label = ['age', 'sex', 'wrat']
y_label = ['FA', 'MD']
degree = 1

# get output data folder
folder_data = pathlib.Path(__file__).parent / 'data'
folder_pdf = pathlib.Path(__file__).parent / 'pdf'
# get data paths
folder = pnl_data.folder_data / 'intrust' / 'fa_md_in_015_NAA_001'
fa_dict = {get_sbj(f.stem): f for f in folder.glob('*FA.nii.gz')}
md_dict = {get_sbj(f.stem): f for f in folder.glob('*MD.nii.gz')}
sbj_mask_dict = {get_sbj(f.stem): f for f in folder.glob('*mask.nii.gz')}

# build sbj_mask_dict
sbj_img_tree = defaultdict(dict)
for d, label in ((fa_dict, 'FA'),
                 (md_dict, 'MD')):
    for sbj, f in d.items():
        sbj_img_tree[sbj][label] = f

# some sbj must be removed because they are missing demographics
sbj_exclude = [get_sbj('016_NA3_007')]
for sbj in sbj_exclude:
    del sbj_img_tree[sbj]

while len(sbj_img_tree) > n_sbj:
    sbj = choice(list(sbj_img_tree.keys()))
    del sbj_img_tree[sbj]

# build regressor
r = PolyRegressor(sbj_img_tree=sbj_img_tree, y_label=y_label, x_label=x_label,
                  sbj_mask_dict=sbj_mask_dict, degree=degree)
r.fit(verbose=True, obs_to_var_thresh=3)

# save
f_out = folder_data / 'poly_regress.p.gz'
file.save(r, f_out)
# r = file.load(f_out)

f_ex_reg = folder_pdf / 'poly_regress_ex.pdf'
with PdfPages(f_ex_reg) as pdf:
    ijk = next(iter(r.ijk_regress_dict.keys()))
    r.plot(ijk, x_feat='age', y_feat='FA')
    plt.suptitle(f'voxel: {ijk}, r2: {r.r2_score[ijk]}')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    pdf.savefig(fig)
    plt.close()

f_r2_image = folder_data / 'poly_regress_r2.nii.gz'
r.r2_to_nii(f_r2_image)

f_r2_hist = folder_pdf / 'poly_regress_r2_hist.pdf'
with PdfPages(f_r2_hist) as pdf:
    r2 = [r.r2_score[ijk] for ijk in r.ijk_regress_dict.keys()]
    plt.hist(r2, bins=100)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    pdf.savefig(fig)
    plt.close()
