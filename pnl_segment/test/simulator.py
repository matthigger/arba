import random
from collections import defaultdict
from datetime import datetime

import numpy as np

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder as folder_data
from pnl_data.set.cidar_post import get_name
from pnl_segment.adaptive.data import FileTree
from pnl_segment.simulate import simulator

rep_per_snr = 10
snr_vec = np.logspace(-2, 1, 20)
p_effect = .5
effect_rad = 3
active_rad = 3
obj = 'max_maha'
feat_list = ['fa', 'md']
f_rba = folder_data / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
folder = folder_data / '2018_Oct_29_10_32AM20'
# folder = None

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

    # only test two sbj
    for sbj in list(sbj_feat_file_tree.keys())[2:]:
        del sbj_feat_file_tree[sbj]

    # init simulator, split into groups
    file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, verbose=True)
    sim = simulator.Simulator(file_tree=file_tree, folder=folder)
    sim.split(p_effect=p_effect)

    file.save(sim, folder / 'sim.p.gz')
else:
    sim = file.load(folder / 'sim.p.gz')

# sim.run_healthy(obj=obj, verbose=True)

arg_list = list()
for snr in snr_vec:
    for _ in range(rep_per_snr):
        arg_list.append({'snr': snr,
                         'obj': obj,
                         'active_rad': active_rad,
                         'radius': effect_rad,
                         'f_rba': f_rba})

parallel.run_par_fnc('run_effect', arg_list=arg_list, obj=sim)
