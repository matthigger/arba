import pathlib
import shutil
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arba.data import SynthFileTree
from arba.effect import get_effect_list, get_sens_spec
from arba.permute import PermuteRegressVBA
from arba.plot import save_fig
from pnl_data.set import hcp_100

# detection params
par_flag = True
num_perm = 96
alpha = .05

# regression effect params
r2_vec = np.logspace(-2, -.5, 21)
# r2_vec = [.9]
num_eff = 10
dim_sbj = 1
num_sbj = 100

str_img_data = 'hcp100'  # 'hcp100' or 'synth'

mask_radius = 6

effect_num_vox = 50

# build dummy folder
folder = pathlib.Path(tempfile.mkdtemp())
shutil.copy(__file__, folder / 'regress_ex_toy.py')
print(folder)

# duild bummy images
if str_img_data == 'synth':
    dim_img = 1
    shape = 6, 6, 6
    file_tree = SynthFileTree(num_sbj=num_sbj, shape=shape,
                              mu=np.zeros(dim_img),
                              cov=np.eye(dim_img), folder=folder / 'raw_data')
elif str_img_data == 'hcp100':
    low_res = True,
    feat_tuple = 'fa',
    file_tree = hcp_100.get_file_tree(lim_sbj=num_sbj,
                                      low_res=low_res,
                                      feat_tuple=feat_tuple)

with file_tree.loaded():
    feat_sbj, eff_list = get_effect_list(file_tree=file_tree,
                                         effect_num_vox=effect_num_vox,
                                         r2=r2_vec[0],
                                         rand_seed=1,
                                         dim_sbj=dim_sbj,
                                         num_eff=num_eff,
                                         no_edge=True,
                                         min_var_mask=True)
    file_tree.to_nii(folder, mean_flag=True)

method_r2ss_list_dict = defaultdict(list)
for eff_idx, eff in enumerate(eff_list):
    for r2 in r2_vec:
        eff = eff.scale_r2(r2)
        file_tree.mask = eff.mask.dilate(mask_radius)
        with file_tree.loaded(effect_list=[eff]):
            perm_reg = PermuteRegressVBA(feat_sbj, file_tree,
                                         num_perm=num_perm,
                                         par_flag=par_flag,
                                         alpha=alpha,
                                         mask_target=eff.mask,
                                         verbose=True,
                                         save_flag=True,
                                         folder=_folder)
            _folder = folder / f'eff{eff_idx}_r2_{r2:.2e}'
            estimate_dict = {'arba': perm_reg.mask_estimate}
            estimate_dict.update(perm_reg.vba_mask_estimate_dict)

            for method, estimate in estimate_dict.items():
                sens, spec = get_sens_spec(target=eff.mask, estimate=estimate,
                                           mask=file_tree.mask)
                s = f'{method} (r2: {r2:.2e}): sens {sens:.3f} spec {spec:.3f}'
                print(s)
                method_r2ss_list_dict[method].append((r2, sens, spec))

sns.set(font_scale=1)
fig, ax = plt.subplots(1, 2)
for idx, feat in enumerate(('sensitivity', 'specificity')):
    plt.sca(ax[idx])
    for method, r2ss_list in method_r2ss_list_dict.items():
        r2 = [x[0] for x in r2ss_list]
        vals = [x[1 + idx] for x in r2ss_list]
        plt.scatter(r2, vals, label=method, alpha=.4)

        d = defaultdict(list)
        for _r2, val in zip(r2, vals):
            d[_r2].append(val)

        x = []
        y = []
        for _r2, val_list in sorted(d.items()):
            x.append(_r2)
            y.append(np.mean(val_list))
        plt.plot(x, y)

    plt.ylabel(feat)
    plt.xlabel(r'$r^2$')
    plt.legend()
    plt.gca().set_xscale('log')
save_fig(folder / 'r2_vs_sens_spec.pdf', size_inches=(10, 4))

print(folder)
