import shutil
from collections import defaultdict

from arba.data import FileTree, scale_normalize, Split
from arba.permute import run_print_single
from arba.seg_graph import SegGraphHistT2
from arba.space import Mask, sample_mask
from pnl_data.set.ofer_tbss import folder

folder_data = folder / 'First_Episode'
grp_tuple = 'fes', 'hc'
feat_tuple = 'fw',
verbose = True
quick = True

# build folder
grp_str = '-'.join(grp_tuple)
feat_str = '-'.join(feat_tuple)
folder_arba = folder / f'arba_{grp_str}_{feat_str}_single'
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

mask = Mask.from_nii(str(folder_data / 'mean_FA_above_p2.nii.gz'))

if quick:
    mask = sample_mask(prior_array=mask, num_vox=1000, ref=mask.ref)

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
with file_tree.loaded():
    sg_hist = SegGraphHistT2(file_tree=file_tree, split=split)
    run_print_single(sg_hist=sg_hist, verbose=verbose, folder=folder_arba)
