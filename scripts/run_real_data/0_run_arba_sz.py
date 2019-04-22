import pathlib
from collections import defaultdict

import numpy as np

from arba.file_tree import FileTree, scale_normalize
from arba.seg_graph import PermuteARBA
from arba.simulate.tfce import PermuteTFCE
from arba.space import Mask
from pnl_data.set.hcp_100 import folder, people

feat_tuple = ('fat', 'fw')
grp_tuple = ('HC', 'SCZ')
n = 100
verbose = True
par_flag = True

# build folder
# folder_arba = pathlib.Path(tempfile.mkdtemp(suffix='_arba'))
# folder_arba.mkdir(exist_ok=True)
folder_arba = pathlib.Path('/tmp/tmp5vdd1ge6_arba')
print(f'folder_arba: {folder_arba}')

# load mask
folder_data = folder / 'low_res'
f_mask = folder_data / 'fat' / 'fat_skel_point5.nii.gz'
mask = Mask.from_nii(f_mask)

# get files per feature
sbj_feat_file_tree = defaultdict(dict)
for p in people:
    if p.grp not in grp_tuple:
        continue

    if not (20 <= p.age <= 30):
        continue

    for feat in feat_tuple:
        f = folder_data / feat.lower() / f'{p}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        sbj_feat_file_tree[p][feat] = f

# limit sbjs to 10 per population
sbj0 = [sbj for sbj in sbj_feat_file_tree.keys() if sbj.grp == grp_tuple[0]][
       :10]
sbj1 = [sbj for sbj in sbj_feat_file_tree.keys() if sbj.grp == grp_tuple[1]][
       :10]
sbj_limit = sbj0 + sbj1
sbj_feat_file_tree = {sbj: d for sbj, d in sbj_feat_file_tree.items()
                      if sbj in sbj_limit}

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
                     fnc_list=[scale_normalize])
split = np.array([sbj.grp == grp_tuple[1] for sbj in file_tree.sbj_list])

# make a smaller test case
from skimage.morphology import binary_dilation
import random
import numpy as np
from arba.space import PointCloud

i, j, k = random.choice(list(file_tree.pc))
_mask = np.zeros(file_tree.ref.shape)
_mask[i, j, k] = 1
for _ in range(20):
    _mask = binary_dilation(_mask)
file_tree.mask = np.logical_and(file_tree.mask, _mask)
file_tree.pc = PointCloud.from_mask(file_tree.mask)

# file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
#                      fnc_list=[scale_normalize])
# split = np.array([sbj.grp == grp_tuple[1] for sbj in file_tree.sbj_list])

# run
perm_arba = PermuteARBA(file_tree, folder=folder_arba)
perm_arba.run(split, n=n, folder=folder_arba, verbose=verbose,
              par_flag=par_flag)

perm_tfce = PermuteTFCE(file_tree, folder=folder_arba)
perm_tfce.run(split, n=n, folder=folder_arba, verbose=verbose,
              par_flag=par_flag)
