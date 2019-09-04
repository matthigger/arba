import shutil

import numpy as np

from arba.effect import get_effect_list, get_sens_spec, EffectRegress
from arba.permute import PermuteRegressVBA
import tempfile
import pathlib

# detection params
par_flag = True
num_perm = 24
alpha = .05

# regression effect params
shape = 5, 5, 5
r2_vec = np.logspace(-2, -.1, 5)
num_eff = 3

effect_num_vox = 10

feat_sbj, file_tree, eff_list = get_effect_list(effect_num_vox=effect_num_vox,
                                                shape=shape, r2=r2_vec[0],
                                                rand_seed=1,
                                                num_eff=num_eff)

folder = pathlib.Path(tempfile.mkdtemp())
shutil.copy(__file__, folder / 'regress_ex_toy.py')

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

        for label, estimate in estimate_dict.items():
            sens, spec = get_sens_spec(target=eff.mask, estimate=estimate,
                                       mask=file_tree.mask)
            print(f'{label} (r2: {r2:.2e}): sens {sens:.3f} spec {spec:.3f}')

print(perm_reg.folder)
