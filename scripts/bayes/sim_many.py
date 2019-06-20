import random
from uuid import uuid4

import numpy as np
from scipy.ndimage.morphology import binary_dilation

import arba
from arba.simulate.sim import run
from pnl_data.set.hcp_100 import folder as folder_hcp
from pnl_data.set.hcp_100 import get_file_tree

# file tree
num_vox = 200
t2 = 0
n_dilate = 5

# build file tree
ft = get_file_tree(low_res=True)
f_rba = folder_hcp / 'to_100307_low_res/100307aparc+aseg.nii.gz'


def get_mask(file_tree, num_vox, n_dilate):
    eff_mask = arba.space.sample_mask_min_var(num_vox=num_vox,
                                              file_tree=file_tree)
    mask = binary_dilation(eff_mask, iterations=n_dilate)
    mask = np.logical_and(mask, ft.mask)

    return mask


label = f'null_num_vox{num_vox}_dilate{n_dilate}'
with ft.loaded():
    grp_split_idx = int(len(ft) / 2)
    arba.data.Split.fix_order(ft.sbj_list)

    while True:
        _folder = folder_hcp / label / str(uuid4())[:5]
        _folder.mkdir(exist_ok=True, parents=True)

        # get random split
        _sbj_list = random.sample(ft.sbj_list, len(ft.sbj_list))
        split = arba.data.Split({'grp_null': _sbj_list[:grp_split_idx],
                                 'grp_eff': _sbj_list[grp_split_idx:]})

        # get random mask
        ft.mask = get_mask(ft, num_vox, n_dilate)

        # run experiment
        run(file_tree=ft, split=split, folder=_folder, verbose=True,
            f_rba=f_rba)
