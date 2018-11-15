import random
from collections import defaultdict
from datetime import datetime

import numpy as np

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder as folder_data
from pnl_data.set.cidar_post import get_name
from pnl_segment.seg_graph.data import FileTree
from pnl_segment.simulate import simulator

num_locations = 2
snr_vec = np.logspace(-2, 2, 5)
# snr_vec = [1]
p_effect = .5
effect_rad = 4
active_rad = 4
obj = 'maha'
feat_list = ['fa', 'md']
effect_u = np.array([1, 0])
resample = False
par_flag = False
f_rba = folder_data / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
# folder = folder_data / '2018_Nov_12_08_16AM31'
folder = None

np.random.seed(1)
random.seed(1)

if folder is None:
    # make output folder (timestamped)
    folder = folder_data / datetime.now().strftime("%Y_%b_%d_%I_%M%p%S")
    folder.mkdir(exist_ok=True, parents=True)

    # build sbj_feat_file_tree
    _folder_data = folder_data / 'dti_in_01193'
    sbj_feat_file_tree = defaultdict(dict)
    for feat in feat_list:
        for f in _folder_data.glob(f'*{feat}.nii.gz'):
            sbj_feat_file_tree[get_name(f.stem)][feat] = f

    # # only test 4 sbj
    # for sbj in list(sbj_feat_file_tree.keys())[4:]:
    #     del sbj_feat_file_tree[sbj]

    # init simulator, split into groups
    file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree)
    sim = simulator.Simulator(file_tree=file_tree, folder=folder, verbose=True,
                              p_effect=p_effect)

    print('begin save')
    file.save(sim, folder / 'sim.p.gz')
    print('end save')
else:
    sim = file.load(folder / 'sim.p.gz')
    sim.folder = folder

# sim.run_effect(snr=100, obj=obj, verbose=True, f_rba=f_rba)
# sim.run_healthy(obj=obj, verbose=True, f_rba=f_rba)

effect_mask_list = [sim.sample_effect_mask(radius=effect_rad)
                    for _ in range(num_locations)]
arg_list = list()
for snr in snr_vec:
    for effect_mask in effect_mask_list:
        d = {'snr': snr,
             'obj': obj,
             'active_rad': active_rad,
             'effect_mask': effect_mask,
             'f_rba': f_rba,
             'resample': resample,
             'u': effect_u}
        arg_list.append(d)
        if not par_flag:
            sim.run_effect(verbose=True, **d)

if par_flag:
    parallel.run_par_fnc('run_effect', arg_list=arg_list, obj=sim)
