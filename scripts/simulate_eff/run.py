from collections import defaultdict

import numpy as np

from arba.data import FileTree, scale_normalize
from arba.plot import plot_performance
from arba.simulate import Simulator
from mh_pytools import file
from pnl_data.set.hcp_100 import get_name, folder

#####################################################################
t2_list = np.logspace(-1, 1, 2)
# t2_list = [10]

num_vox = [50] * 1
# num_vox = np.geomspace(10, 5000, num=11).astype(int)
# num_vox = list(num_vox) * 100
x_axis = 'T-squared'

active_rad = 5
feat_list = ['FA', 'MD']
par_flag = True
effect_shape = 'cube'  # either 'cube' or 'min_var'
num_perm = 5
tfce_flag = False
vba_flag = False

s_feat = '-'.join(feat_list)
folder_out = folder / 'result' / f'ward_test_cube_det'
# shutil.rmtree(str(folder_out))
folder_out.mkdir(exist_ok=True, parents=True)

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

file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree,
                     fnc_list=[scale_normalize])

#####################################################################
# init simulator
with file_tree.loaded():
    sim = Simulator(file_tree=file_tree, folder=folder_out, par_flag=par_flag,
                    effect_shape=effect_shape, num_perm=num_perm,
                    tfce_flag=tfce_flag, vba_flag=vba_flag,
                    active_rad=active_rad)

    # build effects
    sim.build_effect_list(num_vox=num_vox)

    # save
    file.save(sim, folder_out / 'sim.p.gz')

    # run arba
    sim.run(t2_list)

    # run comparison methods
    sim = file.load(folder_out / 'sim.p.gz')
    f_performance = sim.get_performance()

plot_performance(f_performance, x_axis=x_axis)
