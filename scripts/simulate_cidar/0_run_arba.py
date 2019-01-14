from collections import defaultdict
from datetime import datetime

import numpy as np

from mh_pytools import file
from pnl_data.set.cidar_post import get_name, folder
from pnl_segment.seg_graph.data import FileTree
from pnl_segment.simulate import simulator

num_locations = 1
maha_list = np.logspace(-1, 1, 3)
# maha_list = [0]
effect_rad = 5
active_rad = 5
feat_list = ['fa', 'md']
harmonize = True
par_flag = False
light_memory = True

# make output folder (timestamped)
folder_out = folder / datetime.now().strftime("%Y_%b_%d_%H_%M_%S")

# build sbj_feat_file_tree
_folder_data = folder / 'dti_in_01193'
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_list:
    for f in _folder_data.glob(f'*{feat}.nii.gz'):
        sbj_feat_file_tree[get_name(f.stem)][feat] = f

# init simulator, split into groups
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree)
sim = simulator.Simulator(file_tree=file_tree, folder=folder_out)
sim.build_effect_list(n_effect=num_locations, effect_rad=effect_rad,
                      verbose=True)

file.save(sim, folder_out / 'sim.p.gz')

sim.run(maha_list, par_flag=par_flag, active_rad=active_rad,
        harmonize=harmonize, light_memory=light_memory)
