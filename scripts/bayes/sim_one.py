import pathlib
import shutil
import tempfile

import numpy as np
from scipy.ndimage.morphology import binary_dilation

import arba
from pnl_data.set.hcp_100 import get_file_tree, folder
from arba.simulate.sim import run

# file tree
n_sbj = 10
mu = (0, 0)
cov = np.eye(2) * 1
shape = 10, 10, 10

offset = (1, 1)

fake_data = False
one_vs_many = False

np.random.seed(1)

# build file tree
folder_hcp = folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)
folder.mkdir(exist_ok=True, parents=True)

if fake_data:
    # build file tree
    _folder = folder / 'data_orig'
    _folder.mkdir(exist_ok=True, parents=True)
    ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                                 folder=_folder)

    # effect
    mask = np.zeros(shape)
    mask[0:5, 0:5, 0:5] = 1
    mask = arba.space.Mask(mask, ref=ft.ref)
    effect = arba.simulate.Effect(mask=mask, offset=offset)
else:
    # get file tree
    ft = get_file_tree(lim_sbj=10, low_res=True)

    f_rba = folder_hcp / 'to_100307_low_res/100307aparc+aseg.nii.gz'

    # build effect
    with ft.loaded():
        eff_mask = arba.space.sample_mask(ft.mask, num_vox=200, ref=ft.ref)
        fs = ft.get_fs(mask=eff_mask)
        effect = arba.simulate.Effect.from_fs_t2(fs=fs, t2=1,
                                                 mask=eff_mask,
                                                 u=offset)

    # mask file_tree (lowers compute needed)
    mask = binary_dilation(eff_mask, iterations=4)
    ft.mask = np.logical_and(mask, ft.mask)

# build 'split', defines sbj which are affected
if one_vs_many:
    grp_split_idx = n_sbj - 1
else:
    grp_split_idx = int(n_sbj / 2)

arba.data.Split.fix_order(ft.sbj_list)
split = arba.data.Split({'grp_null': ft.sbj_list[:grp_split_idx],
                         'grp_eff': ft.sbj_list[grp_split_idx:]})

# copy script to output folder
f = pathlib.Path(__file__)
shutil.copy(__file__, str(folder / f.name))

split_eff_grp = (split, effect, 'grp_eff')

run(file_tree=ft, split_eff_grp_list=split_eff_grp, folder=folder, verbose=True,
    f_rba=f_rba, print_per_sbj=True, print_hier_seg=True,
    print_lower_bnd=True, print_eff_zoom=True)
