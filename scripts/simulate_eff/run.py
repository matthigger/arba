from collections import defaultdict

import numpy as np

from arba.plot import plot_performance
from arba.seg_graph import FileTree
from arba.simulate import simulator
from mh_pytools import file
from pnl_data.set.hcp_100 import get_name, folder

#####################################################################
# t2_list = np.logspace(-1, 1, 9)
t2_list = [1]

num_vox = [250]
# num_vox = np.geomspace(10, 5000, num=11).astype(int)
# num_vox = list(num_vox) * 100
x_axis = 'T-squared'

active_rad = 5
feat_list = ['FA']
par_flag = True
effect_shape = 'cube'  # either 'cube' or 'min_var'
tfce_num_perm = 200

s_feat = '-'.join(feat_list)
folder_out = folder / 'profile'
# folder_out = folder / f'a_1000_{s_feat}_{effect_shape}'
folder_data = folder / 'to_100307_low_res'
#####################################################################
# build file_tree
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_list:
    for f in folder_data.glob(f'*{feat}.nii.gz'):
        name = get_name(f.stem)
        if name is None:
            # file doesnt fit pattern
            continue
        sbj_feat_file_tree[get_name(f.stem)][feat] = f
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree)

#####################################################################
# init simulator
sim = simulator.Simulator(file_tree=file_tree, folder=folder_out,
                          par_flag=par_flag, effect_shape=effect_shape,
                          tfce_num_perm=tfce_num_perm)

# build effects
sim.build_effect_list(num_vox=num_vox)

# save
file.save(sim, folder_out / 'sim.p.gz')

# run arba
sim.run(t2_list, active_rad=active_rad)

# run comparison methods
sim.run_effect_comparison()

# plot performance
plot_performance(sim.folder / sim.f_performance, x_axis=x_axis)