import os
from collections import defaultdict

from pnl_data.set import ofer_tbss
from arba.seg_graph import FileTree, run_arba
from arba.space import Mask

# params
alpha = .05
active_feat = ['FW', 'FAt']

# output folder
folder_out = ofer_tbss.folder / 'arba'
folder_out.mkdir(exist_ok=True)

# file params
folder = ofer_tbss.folder / 'skel'
f_mask = ofer_tbss.folder / 'stats' / 'mean_FA_skeleton_mask.nii.gz'
feat_sglob_dict = {'FW': '*FW*.nii.gz',
                   'FA': '*FA*.nii.gz',
                   'FAt': '*FAt*.nii.gz'}

# get files per feature
grp_sbj_feat_file_tree = defaultdict(lambda: defaultdict(dict))
for feat in active_feat:
    sglob = feat_sglob_dict[feat]
    for f in folder.glob(sglob):
        # kludge:
        # sbj = ofer_tbss.get_sbj(f.name)[0]
        x = str(f.stem).replace('.', '_').split('_')
        sbj_idx = x[2]
        sbj_grp = x[1].lower()
        sbj = f'{sbj_grp}_{sbj_idx}'

        grp_sbj_feat_file_tree[sbj_grp][sbj][feat] = f

# ensure we found all needed files
for sbj_feat_file_tree in grp_sbj_feat_file_tree.values():
    for feat_f_dict in sbj_feat_file_tree.values():
        if len(feat_f_dict) != len(active_feat):
            raise AttributeError('at least one feature missing')

# build file tree dict
ft_dict = {grp: FileTree(sbj_feat_file_tree)
           for grp, sbj_feat_file_tree in grp_sbj_feat_file_tree.items()}

# load mask, make symlink in output folder
f_mask_sym = folder_out / 'wm_skeleton.nii.gz'
os.remove(str(f_mask_sym))
os.symlink(str(f_mask), f_mask_sym)
mask = Mask.from_nii(f_mask_sym)

# run arba
run_arba(ft_dict, mask, folder_save=folder_out, verbose=True, alpha=alpha)
