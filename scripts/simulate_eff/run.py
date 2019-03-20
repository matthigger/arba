import shutil
from collections import defaultdict

import numpy as np

from arba.data import FileTree, scale_normalize
from arba.plot import plot_performance
from arba.simulate import Simulator
from mh_pytools import file
from pnl_data.set.hcp_100 import get_name, folder

#####################################################################
t2_list = np.logspace(-1, 1, 4)
# t2_list = [.1]

num_vox = [50] * 6
# num_vox = np.geomspace(10, 5000, num=11).astype(int)
# num_vox = list(num_vox) * 100
x_axis = 'T-squared'

active_rad = 5
feat_list = ['FA', 'MD']
par_flag = True
effect_shape = 'cube'  # either 'cube' or 'min_var'
num_perm = 3
tfce_flag = True

print_tree = True
print_image = True
print_hist = True

# make output folder
n = len(t2_list) * len(num_vox)
s_feat = ''.join(feat_list)
folder_out = folder / 'result' / f'{n}_{s_feat}_{effect_shape}'
if folder_out.exists():
    shutil.rmtree(str(folder_out))
folder_out.mkdir(exist_ok=True, parents=True)

#####################################################################
# build file_tree
folder_data = folder / 'to_100307_low_res'
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_list:
    for f in folder_data.glob(f'*{feat}.nii.gz'):
        name = get_name(f.stem)
        if name is None:
            # file doesnt fit pattern
            continue
        sbj_feat_file_tree[get_name(f.stem)][feat] = f

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree,
                     fnc_list=[scale_normalize])

#####################################################################
# init simulator
with file_tree.loaded():
    sim = Simulator(file_tree=file_tree, folder=folder_out, par_flag=par_flag,
                    effect_shape=effect_shape, num_perm=num_perm,
                    tfce_flag=tfce_flag, active_rad=active_rad)

    # build effects
    sim.build_effect_list(num_vox=num_vox)

    # save
    file.save(sim, folder_out / 'sim.p.gz')

    # run arba
    sim.run(t2_list, print_tree=print_tree, print_image=print_image,
            print_hist=print_hist)

    # run comparison methods
    f_performance = sim.get_performance()

plot_performance(f_performance, x_axis=x_axis)
