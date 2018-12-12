import random

import numpy as np

from pnl_data.set.cidar_post import folder as folder_data
from pnl_segment.seg_graph import FeatStat
from pnl_segment.simulate import simulator, Model
from pnl_segment.space import RefSpace, Mask

par_flag = False

sim_per_snr = 1
n_img = 40
snr_vec = np.logspace(-1, 1, 5)
# snr_vec = [1]
p_effect = .5
effect_rad = 3
shape = (9, 9, 9)
obj = 'maha'
effect_u = np.array([-1])
folder = folder_data / 'synth_data'

np.random.seed(1)
random.seed(1)

# build model of 'healthy' (standard normal data)
fs = FeatStat(n=10, mu=0, cov=1)
ijk_fs_dict = {ijk: fs for ijk in np.ndindex(shape)}
model = Model(ijk_fs_dict, shape=shape)

# build file tree
ref = RefSpace(affine=np.eye(4), shape=shape)
file_tree = model.to_file_tree(n=n_img, folder=folder / 'data', ref=ref)

# build sim object
sim = simulator.Simulator(file_tree=file_tree, folder=folder, verbose=True,
                          p_effect=p_effect)

# build effect_mask
ijk_center = np.array(shape) / 2
effect_mask = np.zeros(shape)
for ijk in ijk_fs_dict:
    if np.linalg.norm(ijk_center - np.array(ijk)) < effect_rad:
        effect_mask[ijk] = 1
effect_mask = Mask(effect_mask, ref)

# simulate
arg_list = list()
for snr in snr_vec:
    for _ in range(sim_per_snr):
        d = {'snr': snr,
             'obj': obj,
             'effect_mask': effect_mask,
             'u': effect_u,
             'cross_val_pval': True}
        arg_list.append(d)
        if not par_flag:
            sim.run_effect(verbose=True, **d)

if par_flag:
    raise NotImplementedError
    parallel.run_par_fnc('run_effect', arg_list=arg_list, obj=sim)
