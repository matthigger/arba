from collections import defaultdict

from arba.seg_graph import FileTree, run_arba_cv
from arba.space import Mask
from pnl_data.set.sz import folder, people

# arba params
alpha = .05
feat_tuple = ('fat', 'fw')
grp_tuple = ('HC', 'SCZ')

# run params
par_flag = True
verbose = True

# output folder
s_feat = '-'.join(feat_tuple)
s_grp = '-'.join(grp_tuple)
folder_out = folder / f'arba_cv_{s_grp}_{s_feat}_wm_skel'
folder_out.mkdir(exist_ok=True)

folder_data = folder / 'low_res'

f_mask = folder_data / 'fat' / 'fat_skel_point5.nii.gz'
mask = Mask.from_nii(f_mask)

print(f'folder_out: {folder_out}')
print(f'folder_data: {folder_data}')

# get files per feature
grp_sbj_feat_file_tree = defaultdict(lambda: defaultdict(dict))
for p in people:
    if p.grp not in grp_tuple:
        continue

    if not (20 <= p.age <= 30):
        continue

    for feat in feat_tuple:
        f = folder_data / feat.lower() / f'{p}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        grp_sbj_feat_file_tree[p.grp][p][feat] = f

# build file tree dict
ft_dict = {grp: FileTree(sbj_feat_file_tree)
           for grp, sbj_feat_file_tree in grp_sbj_feat_file_tree.items()}

# run arba
run_arba_cv(ft_dict=ft_dict, folder=folder_out, verbose=verbose, alpha=alpha,
            par_flag=par_flag, mask=mask)
