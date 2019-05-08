from collections import defaultdict

from arba.data import FileTree, scale_normalize, Split
from arba.permute import PermuteARBA
from arba.space import Mask
from pnl_data.set.ofer_tbss import folder
import shutil

folder_data = folder / 'First_Episode'
grp_tuple = 'fes', 'hc'
feat_tuple = 'fw',
n = 24
verbose = True
par_flag = True

# build folder
grp_str = '-'.join(grp_tuple)
feat_str = '-'.join(feat_tuple)
folder_arba = folder / f'arba_{grp_str}_{feat_str}'
if folder_arba.exists():
    shutil.rmtree(str(folder_arba))
folder_arba.mkdir()
print(f'folder_arba: {folder_arba}')

# build file tree
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_tuple:
    _folder = folder_data / feat
    for f in _folder.glob('*.nii.gz'):
        sbj = f.stem.replace('.nii.gz', '')
        sbj = '_'.join(sbj.split('_')[1:])
        sbj_feat_file_tree[sbj][feat] = f

mask = Mask.from_nii(str(f))

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
                     fnc_list=[scale_normalize])

# build split
Split.fix_order(file_tree.sbj_list)
split = defaultdict(list)
for sbj in sbj_feat_file_tree.keys():
    grp = sbj.split('_')[0]
    split[grp].append(sbj)
split = Split(split)

# run
permute_arba = PermuteARBA(file_tree, folder=folder_arba)
permute_arba.run(split, n=n, folder=folder_arba, verbose=verbose,
                 par_flag=par_flag, print_image=True)
