import numpy as np

from arba.simulate.tfce import permute_tfce
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

alpha = .05
folder = folder / 'arba_cv_MF_FA-MD'
n = 5000
verbose = True
par_flag = True

ft_dict = file.load(folder / 'save' / 'ft_dict_.p.gz')

# build folder
folder_tfce = folder / 'tfce'
folder_tfce.mkdir(exist_ok=True)

# build data, mask, split and affine
ft0, ft1 = ft_dict.values()
for ft in ft_dict.values():
    ft.load(verbose=verbose, par_flag=par_flag, load_ijk_fs=False)
x = np.concatenate((ft0.data, ft1.data), axis=3)
split = np.hstack((np.zeros(len(ft0)), np.ones(len(ft1))))
affine = ft0.ref.affine
mask = ft0.mask

# # make a smaller test case
# from arba.space import PointCloud
# from skimage.morphology import binary_dilation
# import random
#
# pc = PointCloud.from_mask(mask)
# i, j, k = random.choice(list(pc))
# _mask = np.zeros(mask.shape)
# _mask[i, j, k] = 1
# for _ in range(8):
#     _mask = binary_dilation(_mask)
# mask = np.logical_and(mask, _mask)

# run tfce
tfce_t2, max_tfce_t2, pval = permute_tfce(x=x, mask=mask, split=split, n=n,
                                          par_flag=par_flag, verbose=verbose,
                                          folder=folder_tfce,
                                          affine=affine, additive=True)
