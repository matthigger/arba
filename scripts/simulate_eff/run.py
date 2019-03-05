from collections import defaultdict
from datetime import datetime

import numpy as np

from arba.plot import plot_performance
from arba.seg_graph import FileTree
from arba.simulate import simulator
from mh_pytools import file
from pnl_data.set.hcp_100 import get_name, folder

#####################################################################
t2_list = np.logspace(-1, 1, 3)
# t2_list = [1]

# effect shape: either radius xor num_vox required
radius = None
num_vox = [100] * 8
# num_vox = np.geomspace(10, 5000, num=11).astype(int)
# num_vox = list(num_vox) * 100


active_rad = 3
feat_list = ['FA', 'MD']
harmonize = False
par_flag = True
edge_per_step = None
effect_shape = 'cube'  # either 'cube' or 'min_var'
tfce_num_perm = 100

# folder_data = folder / 'dti_in_01193'
folder_data = folder / 'to_100307_low_res'

x_axis = 'T-squared'
#####################################################################
# make output folder (timestamped)
folder_out = folder / datetime.now().strftime("%Y_%b_%d_%H_%M_%S")

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
sim.build_effect_list(radius=radius, num_vox=num_vox)

# save
file.save(sim, folder_out / 'sim.p.gz')

# run arba
sim.run(t2_list, active_rad=active_rad, harmonize=harmonize,
        edge_per_step=edge_per_step, grp_r2=sim.grp_null)

# run comparison methods
sim.run_effect_comparison()

# plot performance
plot_performance(sim.folder / sim.f_performance, x_axis=x_axis)
