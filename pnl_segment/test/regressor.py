"""     x_label (list): label of each dimension of x
        y_label (list): str label of each dimension of y, note that each key in
                       top level of sbj_img_tree must have each y_label
        sbj_img_tree (dict): 1st level: sbj objects, which have each of y_label
                                        as attributes.
                             2nd level: feat label, must be in x_feat
                             3rd level: (str or Path) to nii image
        sbj_mask_dict (dict): keys are sbj objects, values are str or Path to
                              masks of valid values"""
from collections import defaultdict

import pnl_data
from pnl_data.set.intrust import get_sbj
from pnl_segment.vox_regress.regressor import Regressor

x_label = ['FA', 'MD']
y_label = ['age', 'sex', 'wrat']

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

# temp, delete all but first 4 sbj
sbj_exclude = list(sbj_img_tree.keys())[4:]
for sbj in sbj_exclude:
    del sbj_img_tree[sbj]

# build regressor
r = Regressor(sbj_img_tree=sbj_img_tree, y_label=y_label, x_label=x_label,
              sbj_mask_dict=sbj_mask_dict)
r.learn()
