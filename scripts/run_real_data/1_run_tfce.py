import shutil

from arba.simulate.tfce import PermuteTFCE
from mh_pytools import file
from pnl_data.set.sz import folder

folder = folder / 'arba_cv_HC-FES_fat-fw_wm_skel'
n = 10
verbose = True
par_flag = False

ft_dict = file.load(folder / 'save' / 'ft_dict_.p.gz')

# build folder
folder_tfce = folder / 'tfce'
shutil.rmtree(str(folder_tfce))
folder_tfce.mkdir(exist_ok=True)

print(f'folder_tfce: {folder_tfce}')

# make a smaller test case
from arba.space import PointCloud
from skimage.morphology import binary_dilation
import random
import numpy as np

mask = next(iter(ft_dict.values())).mask
pc = PointCloud.from_mask(mask)
i, j, k = random.choice(list(pc))
_mask = np.zeros(mask.shape)
_mask[i, j, k] = 1
for _ in range(5):
    _mask = binary_dilation(_mask)
mask = np.logical_and(mask, _mask)
for ft in ft_dict.values():
    ft.mask = mask

# run
perm_tfce = PermuteTFCE(ft_dict)
perm_tfce.run(n=n, verbose=verbose, par_flag=par_flag, folder=folder_tfce)
