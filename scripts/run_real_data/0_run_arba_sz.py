import pathlib
import tempfile
from collections import defaultdict

from arba.data import FileTree, scale_normalize, Split
from arba.permute import PermuteARBA
from arba.space import sample_mask, Mask
from pnl_data.set.sz import folder, people

feat_tuple = 'fw',
grp_tuple = ('HC', 'SCZ')
n = 240
verbose = True
par_flag = True

quick = False
tmp_folder = False

# build folder
if tmp_folder:
    folder_arba = pathlib.Path(tempfile.mkdtemp(suffix='_arba'))
else:
    grp_str = '-'.join(grp_tuple)
    feat_str = '-'.join(feat_tuple)
    folder_arba = folder / f'arba_{grp_str}_{feat_str}'
folder_arba.mkdir(exist_ok=True)
print(f'folder_arba: {folder_arba}')

# load mask
folder_data = folder / 'low_res'
mask = Mask.from_nii(folder_data / 'fat' / 'fat_mask_all.nii.gz')

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
    file_tree.mask = sample_mask(prior_array=file_tree.mask,
                                 num_vox=1000,
                                 ref=file_tree.ref)

    # build split with new order
    Split.fix_order(file_tree.sbj_list)
    split = defaultdict(list)
    for sbj in sbj_feat_file_tree.keys():
        split[sbj.grp].append(sbj)
    split = Split(split)

# run
permute_arba = PermuteARBA(file_tree, folder=folder_arba)
permute_arba.run(split, n=n, folder=folder_arba, verbose=verbose,
                 par_flag=par_flag, print_tree=True)
