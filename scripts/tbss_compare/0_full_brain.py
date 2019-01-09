from collections import defaultdict

from pnl_data.set import ofer_tbss
from pnl_segment.seg_graph import FileTree, run_arba

# params
alpha = .05
edge_per_step = .001
active_feat = ['FW', 'FAt']

# output folder
folder_out = ofer_tbss.folder / 'arba_full'
folder_out.mkdir(exist_ok=True)

# file params
feat_sglob_dict = {'FW': '**/FA/*_to_target_FW.nii.gz',
                   'FA': '**/FA/*_to_target_FA.nii.gz',
                   'FAt': '**/FA/*_to_target_FAt.nii.gz'}

# get files per feature
grp_sbj_feat_file_tree = defaultdict(lambda: defaultdict(dict))
for feat in active_feat:
    sglob = feat_sglob_dict[feat]
    for f in ofer_tbss.folder.glob(sglob):
        sbj = ofer_tbss.get_sbj(f.name)[0]
        grp_sbj_feat_file_tree[sbj.grp][sbj][feat] = f

# ensure we found all needed files
for sbj_feat_file_tree in grp_sbj_feat_file_tree.values():
    for feat_f_dict in sbj_feat_file_tree.values():
        if len(feat_f_dict) != len(active_feat):
            raise AttributeError('at least one feature missing')

# build file tree dict
ft_dict = {grp: FileTree(sbj_feat_file_tree)
           for grp, sbj_feat_file_tree in grp_sbj_feat_file_tree.items()}

# run arba
run_arba(ft_dict, folder_save=folder_out, verbose=True, alpha=alpha,
         edge_per_step=edge_per_step)
