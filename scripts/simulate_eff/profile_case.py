import pathlib
import random
import tempfile
from collections import defaultdict

import numpy as np

from arba.file_tree import FileTree, scale_normalize
from arba.plot import save_fig
from arba.plot.scatter_tree import size_v_wt2, size_v_t2
from arba.seg_graph import PermuteARBA
from arba.simulate import Effect
from arba.space import sample_mask, PointCloud
from pnl_data.set.hcp_100 import folder, people

#####################################################################
t2 = 10
num_vox = 250

active_rad = 5
feat_list = ['FA', 'MD']
effect_shape = 'cube'  # either 'cube' or 'min_var'

folder_out = pathlib.Path(tempfile.mkdtemp())
folder_out.mkdir(exist_ok=True, parents=True)
print(f'folder_out: {folder_out}')

folder_data = folder / 'to_100307_low_res'
#####################################################################
# build file_tree
sbj_feat_file_tree = defaultdict(dict)
for p in people:
    for feat in feat_list:
        f = folder_data / f'{p.name}_{feat}.nii.gz'
        assert f.exists(), f'file not found {f}'
        sbj_feat_file_tree[p][feat] = f

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree)

#####################################################################
np.random.seed(1)
random.seed(1)
with file_tree.loaded():
    split = tuple([False] * 50 + [True] * 50)

    mask = sample_mask(prior_array=file_tree.mask, num_vox=num_vox,
                       ref=file_tree.ref)
    pc = PointCloud.from_mask(mask)
    fs = sum(file_tree.get_fs(ijk) for ijk in pc)
    effect = Effect.from_fs_t2(fs, t2=t2, mask=mask)

    file_tree.split_effect = (split, effect)

    mask_active = mask.dilate(active_rad)
    file_tree.mask = mask_active
    file_tree.pc = PointCloud.from_mask(mask_active)

    ##########

    permuteARBA = PermuteARBA(file_tree, folder=folder_out)
    sg_hist = permuteARBA.run_split(split)

    grp_sbj_dict = {'0': file_tree.sbj_list[:50],
                    '1': file_tree.sbj_list[50:]}
    tree_hist, node_reg_dict = sg_hist.merge_record.resolve_hist(file_tree,
                                                                 grp_sbj_dict)
    size_v_wt2(tree_hist, mask=effect.mask)
    print(save_fig())

    size_v_t2(tree_hist, mask=effect.mask)
    print(save_fig())