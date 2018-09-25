from collections import defaultdict
from random import choice

import pnl_data
from pnl_data.set.intrust import get_sbj
from pnl_segment.vox_regress.polyregressor import PolyRegressor
import pathlib
from mh_pytools import file

n_sbj = 10
x_label = ['FA', 'MD']
y_label = ['age', 'sex', 'wrat']

# get output data folder
folder_data = pathlib.Path(__file__).parent / 'data'

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
                  sbj_mask_dict=sbj_mask_dict)
r.fit(verbose=True)

# save
f_out = folder_data / 'poly_regress.p.gz'
file.save(r, f_out)
