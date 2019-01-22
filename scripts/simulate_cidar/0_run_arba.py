from collections import defaultdict
from datetime import datetime

import nibabel as nib
import numpy as np

from mh_pytools import file
from pnl_data.set.cidar_post import get_name, folder
from pnl_segment.seg_graph.data import FileTree
from pnl_segment.simulate import simulator

#######################
num_locations = 100
maha_list = np.logspace(-1, 1, 9)
# maha_list = [10]

# effect shape, either radius xor num_vox
radius = None
f_seg_array = folder / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
seg_array = nib.load(str(f_seg_array)).get_data()
num_vox = 200

active_rad = 5
feat_list = ['fa', 'md']
harmonize = True
par_flag = True
#######################

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
sim.build_effect_list(n_effect=num_locations, radius=radius, verbose=True,
                      seg_array=seg_array, num_vox=num_vox)

file.save(sim, folder_out / 'sim.p.gz')

sim.run(maha_list, par_flag=par_flag, active_rad=active_rad,
        harmonize=harmonize)
