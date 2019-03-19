import random
import shutil
from collections import defaultdict

import numpy as np

import pnl_data
from arba.data import FileTree
from arba.seg_graph import PermuteARBA, FeatStat
from arba.simulate import Model
from arba.space import RefSpace, Mask

alpha = .05
n_img = 100
feat_var = 1
shape = (4, 4, 1)
# average feature for bottom, top of image in each population
pop_mu_bt_dict = {'pop0': (0, 0),
                  'pop1': (10, 0)}
folder = pnl_data.folder_data / 'arba_toy_ex'
shutil.rmtree(str(folder))
folder.mkdir(exist_ok=True)

# init random seed
np.random.seed(1)
random.seed(1)

# build dummy ref space
ref = RefSpace(affine=np.eye(4), shape=shape)

# build model of each population
model_dict = dict()
ft_dict = dict()
mask_effect = Mask(np.zeros(shape), ref=ref)
mid_point_j = shape[1] / 2
sbj_feat_file_tree = defaultdict(dict)
grp_sbj_dict = defaultdict(list)
for pop, (mu_btm, mu_top) in pop_mu_bt_dict.items():
    ijk_fs_dict = dict()
    for ijk in np.ndindex(shape):
        if ijk[1] < mid_point_j:
            # bottom of image
            ijk_fs_dict[ijk] = FeatStat(n=1000, mu=mu_btm, cov=feat_var)
            mask_effect[ijk] = 1
        else:
            # top of image
            ijk_fs_dict[ijk] = FeatStat(n=1000, mu=mu_top, cov=feat_var)

    model = Model(ijk_fs_dict, shape=shape)
    for idx in range(n_img):
        sbj = f'{pop}_{idx}'
        sbj_feat_file_tree[sbj]['feat'] = model.sample_nii()
        grp_sbj_dict[pop].append(sbj)

# build file tree
file_tree = FileTree(sbj_feat_file_tree)
sbj_list = [f'pop1_{idx}' for idx in range(n_img)]
split = file_tree.sbj_list_to_bool(sbj_list)

mask_effect.to_nii(folder / 'mask_effect.nii.gz')

# build mask
mask = Mask(np.ones(shape), ref=ref)

# run arba
with file_tree.loaded():
    permuteARBA = PermuteARBA(file_tree)
    sg_hist = permuteARBA.run_split(split, full_t2=True)
    permuteARBA.run(split, n=100, folder=folder, print_image=True,
                    print_tree=True)

    print(folder)
