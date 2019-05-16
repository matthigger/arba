import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from tqdm import tqdm

from arba.data import FileTree, Split
from arba.permute import run_single, print_seg
from arba.plot import save_fig
from arba.seg_graph import SegGraphHistT2
from arba.simulate import Effect
from arba.space import Mask, sample_mask
from mh_pytools import file
from pnl_data.set import hcp_100

feat_tuple = 'fa', 'md'
num_vox = 5000
verbose = True
eff_num_vox_edge_n_list = [(750, None),
                           (250, None),
                           (750, 3)]
eff_t2 = np.logspace(-2, 1, 21)
eff_u = (-1, 0)

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
            eff = Effect.from_fs_t2(fs, t2=1, mask=eff_mask, edge_n=eff_edge_n,
                                    u=eff_u)
            eff_list.append(eff)

        # ensure effects do not overlap
        eff_mask_sum = sum(eff.mask for eff in eff_list)
        if eff_mask_sum.max() <= 1:
            break
        else:
            print(f'overlapping effects, resampling (seed was {seed})')
            seed += 1

# test each method
auc_dict = defaultdict(list)
with file_tree.loaded():
    for _t2 in tqdm(eff_t2, desc='per t2'):
        # set output folder
        _folder = folder_out / f't2_{_t2:.1e}'
        _folder.mkdir()

        # scale effects
        for eff_idx, eff in enumerate(eff_list):
            eff.t2 = _t2
            eff.to_nii(f_out=_folder / f'eff_{eff_idx}.nii.gz')
            eff.mask.to_nii(f_out=folder_out / f'eff_mask_{eff_idx}.nii.gz')
        split_eff_list = [(split, eff) for eff in eff_list]

        # apply effects
        str_feat = '_'.join(feat_tuple)
        file_tree.reset_effect(split_eff_list)
        file_tree.to_nii(folder=_folder)
        file_tree.mask.to_nii(f_out=_folder / 'mask.nii.gz')

        # run
        sg_hist = SegGraphHistT2(file_tree=file_tree, split=split)
        sg_hist, folder = run_single(sg_hist=sg_hist, folder=_folder,
                                     verbose=verbose)
        print_seg(sg_hist=sg_hist, folder=folder)
        plt.close('all')

        # compute auc
        img_t2 = nib.load(str(_folder / 't2.nii.gz'))
        t2 = img_t2.get_data()
        for eff in eff_list:
            mask = file_tree.mask - sum(e.mask for e in eff_list) + eff.mask
            _auc = eff.get_auc(t2, mask)
            auc_dict[eff].append(_auc)

auc_dict = dict(auc_dict)
file.save(auc_dict, folder_out / 'auc_dict.p.gz')

# print auc graph
sns.set(font_scale=1.2)
plt.figure()
for (eff, auc), (num_vox, edge_n) in zip(auc_dict.items(),
                                         eff_num_vox_edge_n_list):
    label = f'num_vox={num_vox}, edge_n={edge_n}'
    plt.plot(eff_t2, auc_dict[eff], label=label)
plt.legend()
plt.gca().set_xscale('log')
save_fig(folder_out / 'auc.pdf')
