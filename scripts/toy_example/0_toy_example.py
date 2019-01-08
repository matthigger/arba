import random
import shutil

import numpy as np

import pnl_data
from pnl_segment.seg_graph import FeatStat, run_arba
from pnl_segment.simulate import Model
from pnl_segment.space import RefSpace, Mask

alpha = .05
n_img = 100
feat_var = 1
shape = (4, 4, 1)
# average feature for bottom, top side of image in each population
pop_mu_bt_dict = {'pop0': (0, 0),
                  'pop1': (3, 0)}
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
for pop, (mu_btm, mu_top) in pop_mu_bt_dict.items():
    ijk_fs_dict = dict()
    for ijk in np.ndindex(shape):
        if ijk[1] < mid_point_j:
            # bottom of image
            ijk_fs_dict[ijk] = FeatStat(n=2, mu=mu_btm, cov=feat_var)
            mask_effect[ijk] = 1
        else:
            # top of image
            ijk_fs_dict[ijk] = FeatStat(n=2, mu=mu_top, cov=feat_var)

    model = Model(ijk_fs_dict, shape=shape)

    # build file tree
    ft_dict[pop] = model.to_file_tree(n=n_img,
                                      folder=folder / 'data',
                                      label=f'{pop}_',
                                      ref=ref)

    # store model
    model_dict[pop] = model

mask_effect.to_nii(folder / 'mask_effect.nii.gz')

# build mask
mask = Mask(np.ones(shape), ref=ref)

# run arba
run_arba(ft_dict, mask=mask, folder_save=folder, alpha=alpha, harmonize=False,
         verbose=True)
