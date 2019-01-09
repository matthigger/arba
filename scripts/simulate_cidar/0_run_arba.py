import random
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import get_name, folder
from pnl_segment.seg_graph.data import FileTree
from pnl_segment.simulate import simulator, Effect

num_locations = 10
# maha_vec = np.logspace(-1, 1, 9)
maha_vec = [0]
p_effect = .5
effect_rad = 5
active_rad = 5
feat_list = ['fa', 'md']
effect_u = np.array([-1, 0])
harmonize = True
par_flag = False

# make output folder (timestamped)
folder_out = folder / datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
folder_out.mkdir(exist_ok=True, parents=True)

# build sbj_feat_file_tree
_folder_data = folder / 'dti_in_01193'
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_list:
    for f in _folder_data.glob(f'*{feat}.nii.gz'):
        sbj_feat_file_tree[get_name(f.stem)][feat] = f

# init simulator, split into groups
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree)
sim = simulator.Simulator(file_tree=file_tree, folder=folder_out,
                          p_effect=p_effect)

file.save(sim, folder_out / 'sim.p.gz')

np.random.seed(1)
random.seed(1)

# sample effect locations (constant across maha)
effect_mask_list = list()
for _ in tqdm(range(num_locations), desc='sample effect mask'):
    mask = Effect.sample_mask(prior_array=file_tree.mask,
                              radius=effect_rad,
                              ref=file_tree.ref)
    effect_mask_list.append(mask)

# build arg_list
arg_list = list()
for maha in maha_vec:
    for effect_mask in effect_mask_list:
        d = {'maha': maha,
             'active_rad': active_rad,
             'effect_mask': effect_mask,
             'harmonize': harmonize,
             'u': effect_u,
             'verbose': not par_flag}
        arg_list.append(d)

if par_flag:
    parallel.run_par_fnc('run_effect', arg_list=arg_list, obj=sim)
if not par_flag:
    for d in arg_list:
        sim.run_effect(**d)
