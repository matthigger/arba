import pathlib
import tempfile

import nibabel as nib
import arba
import numpy as np
import os
from mh_pytools import file

# file tree
n_sbj = 10
mu = (0, 0)
cov = np.eye(2)
shape = 10, 10, 10

# effect
mask = np.zeros(shape)
mask[3:7, 3:7, 3:7] = 1
offset = (10, 10)

# build file tree
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
os.mkdir(str(folder))
print(folder)
ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                             folder=folder)

# build effect
effect = arba.simulate.Effect(mask=mask, offset=offset)

# build 'split', defines sbj which are affected
arba.data.Split.fix_order(ft.sbj_list)
half = int(n_sbj / 2)
split = arba.data.Split({False: ft.sbj_list[:half],
                         True: ft.sbj_list[half:]})

# go
with ft.loaded(split_eff_list=[(split, effect)]):
    # write features to block img
    for feat_idx, feat in enumerate(ft.feat_list):
        img = nib.Nifti1Image(ft.data[:, :, :, :, feat_idx], ft.ref.affine)
        img.to_filename(str(folder / f'{feat}.nii.gz'))

    # agglomerative clustering
    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=ft, split=split)
    sg_hist.reduce_to(1)

    # save
    f = folder / 'sg_hist.p.gz'
    file.save(sg_hist, f)