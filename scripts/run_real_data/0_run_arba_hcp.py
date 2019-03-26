from collections import defaultdict

from arba.data import FileTree
from arba.seg_graph import PermuteARBA
from arba.simulate import PermuteTFCE
from pnl_data.set.hcp_100 import folder, people

# arba params
alpha = .05
feat_tuple = 'FA',
num_perm = 1000

# run params
par_flag = True
verbose = True

# output folder
s_feat = '-'.join(feat_tuple)
folder_out = folder / f'arba_cv_MF_{s_feat}'
folder_out.mkdir(exist_ok=True)

folder_data = folder / 'to_100307_low_res'

# get files per feature
sbj_feat_file_tree = defaultdict(dict)
sbj_male_list = list()
for p in people:
    for feat in feat_tuple:
        f = folder_data / f'{p.name}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        sbj_feat_file_tree[p][feat] = f
    if p.gender == 'M':
        sbj_male_list.append(p)

file_tree = FileTree(sbj_feat_file_tree)
split = file_tree.sbj_list_to_bool(sbj_male_list)

# build file tree dict
folder_arba = folder_out / 'arba_permute'
folder_tfce = folder_out / 'tfce'

with file_tree.loaded():
    permute_arba = PermuteARBA(file_tree)
    permute_arba.run(split, par_flag=True, verbose=True, folder=folder_arba,
                     n=num_perm, print_image=True, save_self=True,
                     print_tree=True, print_hist=True, print_region=True)

    permute_tfce = PermuteTFCE(file_tree)
    permute_tfce.run(split, par_flag=True, verbose=True,
                     folder=folder_tfce, n=num_perm, print_image=True,
                     save_self=True, print_hist=True, print_region=True)
