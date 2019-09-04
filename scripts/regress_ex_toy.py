import shutil

import numpy as np

from arba.effect import get_effect_list, get_sens_spec, EffectRegress
from arba.permute import PermuteRegressVBA
from arba.plot import save_fig
import tempfile
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# detection params
par_flag = True
num_perm = 24
alpha = .05

# regression effect params
shape = 6, 6, 6
r2_vec = np.logspace(-2, -.1, 7)
num_eff = 5

effect_num_vox = 27

feat_sbj, file_tree, eff_list = get_effect_list(effect_num_vox=effect_num_vox,
                                                shape=shape, r2=r2_vec[0],
                                                rand_seed=1,
                                                num_eff=num_eff)

folder = pathlib.Path(tempfile.mkdtemp())
shutil.copy(__file__, folder / 'regress_ex_toy.py')
method_r2ss_list_dict = defaultdict(list)
for eff in eff_list:
    for r2 in r2_vec:
        eff = eff.scale_r2(r2)

        with file_tree.loaded(effect_list=[eff]):
            perm_reg = PermuteRegressVBA(feat_sbj, file_tree,
                                         num_perm=num_perm,
                                         par_flag=par_flag,
                                         alpha=alpha,
                                         mask_target=eff.mask,
                                         verbose=True,
                                         save_flag=False)
            estimate_dict = {'arba': perm_reg.mask_estimate}
            estimate_dict.update(perm_reg.vba_mask_estimate_dict)

        for method, estimate in estimate_dict.items():
            sens, spec = get_sens_spec(target=eff.mask, estimate=estimate,
                                       mask=file_tree.mask)
            print(f'{method} (r2: {r2:.2e}): sens {sens:.3f} spec {spec:.3f}')
            method_r2ss_list_dict[method].append((r2, sens, spec))


sns.set(font_scale=1)
fig, ax = plt.subplots(1, 2)
for idx, feat in enumerate(('sensitivity', 'specificity')):
    plt.sca(ax[idx])
    for method, r2ss_list in method_r2ss_list_dict.items():
        r2 = [x[0] for x in r2ss_list]
        vals = [x[1 + idx] for x in r2ss_list]
        plt.scatter(r2, vals, label=method, alpha=.5)

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
