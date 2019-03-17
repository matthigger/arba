import pathlib
import tempfile
from collections import defaultdict

import numpy as np

from arba.data import FileTree, scale_normalize
from arba.simulate import PermuteTFCE
from arba.space import Mask
from pnl_data.set.sz import folder, people

feat_tuple = ('fat', 'fw')
grp_tuple = ('HC', 'SCZ')
n = 100
verbose = True
par_flag = True

# build folder
folder_tfce = pathlib.Path(tempfile.mkdtemp(suffix='_tfce'))
folder_tfce.mkdir(exist_ok=True)

print(f'folder_tfce: {folder_tfce}')

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

# # limit sbjs to 10 per population
# sbj0 = [sbj for sbj in sbj_feat_file_tree.keys() if sbj.grp == grp_tuple[0]][:10]
# sbj1 = [sbj for sbj in sbj_feat_file_tree.keys() if sbj.grp == grp_tuple[1]][:10]
# sbj_limit = sbj0 + sbj1
# sbj_feat_file_tree = {sbj: d for sbj, d in sbj_feat_file_tree.items()
#                       if sbj in sbj_limit}
#
# file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
#                      fnc_list=[scale_normalize])
# split = np.array([sbj.grp == grp_tuple[1] for sbj in file_tree.sbj_list])
#
# # make a smaller test case
# from skimage.morphology import binary_dilation
# import random
# import numpy as np
# from arba.space import PointCloud
#
# i, j, k = random.choice(list(file_tree.pc))
# _mask = np.zeros(file_tree.ref.shape)
# _mask[i, j, k] = 1
# for _ in range(5):
#     _mask = binary_dilation(_mask)
# file_tree.mask = np.logical_and(file_tree.mask, _mask)
# file_tree.pc = PointCloud.from_mask(file_tree.mask)

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask,
                     fnc_list=[scale_normalize])
split = np.array([sbj.grp == grp_tuple[1] for sbj in file_tree.sbj_list])

# run
perm_tfce = PermuteTFCE(file_tree, folder=folder_tfce)
perm_tfce.run(split, n=n, folder=folder_tfce, verbose=verbose,
              par_flag=par_flag)
