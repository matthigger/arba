from collections import defaultdict

from arba.seg_graph import FileTree, run_arba_cv
from pnl_data.set.sz import folder, people

# arba params
alpha = .05
grp_tuple = ('HC', 'SZ')
feat_tuple = ('fa', 'md')
low_res = True

# run params
par_flag = True
verbose = True

# output folder
s_grp = '-'.join(grp_tuple)
s_feat = '-'.join(feat_tuple)
folder_out = folder / f'arba_cv_{s_grp}_{s_feat}'
folder_out.mkdir(exist_ok=True)

folder_data = folder
if low_res:
    folder_data = folder_data / 'low_res'

# get files per feature
grp_sbj_feat_file_tree = defaultdict(lambda: defaultdict(dict))
for p in people:
    if p.grp not in grp_tuple or not (25 <= p.age <= 35):
        continue

    for feat in feat_tuple:
        f = folder_data / feat / f'{p.name}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        grp_sbj_feat_file_tree[p.grp][p][feat] = f

# build file tree dict
ft_dict = {grp: FileTree(sbj_feat_file_tree)
           for grp, sbj_feat_file_tree in grp_sbj_feat_file_tree.items()}

# run arba
run_arba_cv(ft_dict=ft_dict, folder=folder_out, verbose=verbose, alpha=alpha,
            par_flag=par_flag)
