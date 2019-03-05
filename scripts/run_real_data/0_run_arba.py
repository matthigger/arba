from collections import defaultdict

from arba.seg_graph import FileTree, run_arba_cv
from pnl_data.set.hcp_100 import folder, people

# arba params
alpha = .05
feat_tuple = ('FA', 'MD')

# run params
par_flag = True
verbose = True

# output folder
s_feat = '-'.join(feat_tuple)
folder_out = folder / f'arba_cv_MF_{s_feat}'
folder_out.mkdir(exist_ok=True)

folder_data = folder / 'to_100307_low_res'

# get files per feature
grp_sbj_feat_file_tree = defaultdict(lambda: defaultdict(dict))
for p in people:
    for feat in feat_tuple:
        f = folder_data / f'{p.name}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        grp_sbj_feat_file_tree[p.gender][p][feat] = f

# build file tree dict
ft_dict = {grp: FileTree(sbj_feat_file_tree)
           for grp, sbj_feat_file_tree in grp_sbj_feat_file_tree.items()}

# run arba
run_arba_cv(ft_dict=ft_dict, folder=folder_out, verbose=verbose, alpha=alpha,
            par_flag=par_flag)
