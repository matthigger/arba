import nibabel as nib
import numpy as np

import arba

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
ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov)

# build effect
effect = arba.simulate.Effect(mask=mask, offset=offset)

# build 'split', defines sbj which are affected
arba.data.Split.fix_order(ft.sbj_list)
half = int(n_sbj / 2)
split = arba.data.Split({False: ft.sbj_list[:half],
                         True: ft.sbj_list[half:]})

# go
with ft.loaded(split_eff_list=[(split, effect)]):
    img = nib.Nifti1Image(ft.data[:, :, :, :, 0], np.eye(4))
    img.to_filename('/tmp/temp.nii.gz')
