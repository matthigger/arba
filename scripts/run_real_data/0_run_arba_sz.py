import pathlib
import tempfile
from collections import defaultdict

from arba.data import FileTree, scale_normalize, Split
from arba.permute import PermuteARBA
from arba.space import Mask, sample_mask
from pnl_data.set.sz import folder, people

feat_tuple = 'fw',
grp_tuple = ('HC', 'SCZ')
n = 24
verbose = True
par_flag = True

quick = True

# build folder
folder_arba = pathlib.Path(tempfile.mkdtemp(suffix='_arba'))
folder_arba.mkdir(exist_ok=True)
print(f'folder_arba: {folder_arba}')

# load mask
folder_data = folder / 'low_res'
f_mask = folder_data / 'fat' / 'mean_fat.nii.gz'
mask = Mask.from_nii(f_mask)

# build file tree
sbj_feat_file_tree = defaultdict(dict)
for p in people:
    if p.grp not in grp_tuple:
        continue

    if not (20 <= p.age <= 30):
        continue

    for feat in feat_tuple:
        f = folder_data / feat.lower() / f'{p}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        sbj_feat_file_tree[p][feat] = f
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
                     fnc_list=[scale_normalize])

# build split
Split.fix_order(file_tree.sbj_list)
split = defaultdict(list)
for sbj in sbj_feat_file_tree.keys():
    split[sbj.grp].append(sbj)
split = Split(split)

if quick:
    # discard to a few sbj and a small area
    file_tree.discard_to(10, split=split)
    Split.fix_order(file_tree.sbj_list)

    # sample to smaller mask
    mask = sample_mask(prior_array=file_tree.mask,
                       num_vox=200,
                       ref=file_tree.ref)
    file_tree.apply_mask(mask=mask)

# run
permute_arba = PermuteARBA(file_tree, folder=folder_arba)
permute_arba.run(split, n=n, folder=folder_arba, verbose=verbose,
                 par_flag=par_flag)
