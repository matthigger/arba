import random
import shutil
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from arba.data import FileTree, Split
from arba.permute import run_print_single
from arba.seg_graph import SegGraphHistT2
from arba.simulate import Effect
from arba.space import Mask, sample_mask
from pnl_data.set import hcp_100

feat_tuple = 'fa', 'md'
num_vox = int(1e5)

eff_num_vox_edge_n_list = [(1000, None),
                           (200, None),
                           (1000, 3)]
eff_t2 = np.logspace(-1, 1, 11)

# build output dir
folder_out = hcp_100.folder / 'eff_soft'
if folder_out.exists():
    shutil.rmtree(str(folder_out))
folder_out.mkdir()

# build file tree
folder = hcp_100.folder / 'to_100307_low_res'
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_tuple:
    for f in folder.glob(f'*_{feat.upper()}.nii.gz'):
        sbj = f.stem.split('_')[0]
        try:
            int(sbj)
        except ValueError:
            # not a sbj, each sbj is numeric
            continue
        sbj_feat_file_tree[sbj][feat] = f

# build mask, sample to num_vox
mask = Mask.from_nii(f_nii=folder / 'mean_FA.nii.gz')
mask = sample_mask(prior_array=mask,
                   num_vox=num_vox,
                   ref=mask.ref)
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask)

# build split of data
Split.fix_order(file_tree.sbj_list)
n = int(len(file_tree.sbj_list) / 2)
split = Split({True: file_tree.sbj_list[:n],
               False: file_tree.sbj_list[n:]})

# build effect
with file_tree.loaded():
    seed = 1
    while True:
        eff_list = list()
        np.random.seed(seed)
        random.seed(seed)
        for eff_num_vox, eff_edge_n in eff_num_vox_edge_n_list:
            eff_mask = sample_mask(prior_array=mask,
                                   num_vox=eff_num_vox,
                                   ref=mask.ref)
            fs = file_tree.get_fs(mask=eff_mask)
            eff = Effect.from_fs_t2(fs, t2=1, mask=eff_mask, edge_n=eff_edge_n)
            eff_list.append(eff)

        # ensure effects do not overlap
        eff_mask_sum = sum(eff.mask for eff in eff_list)
        if eff_mask_sum.max() <= 1:
            break
        else:
            print(f'overlapping effects, resampling (seed was {seed})')
            seed += 1

# test each method
for _t2 in tqdm(eff_t2, desc='per t2'):
    # scale effects
    for eff in eff_list:
        eff.t2 = _t2
    split_eff_dict = {split: eff for eff in eff_list}

    _folder = folder_out / f't2_{_t2:.2e}'
    _folder.mkdir()

    with file_tree.loaded(split_eff_dict):
        sg_hist = SegGraphHistT2(file_tree=file_tree, split=split)
        run_print_single(sg_hist=sg_hist, folder=_folder)
