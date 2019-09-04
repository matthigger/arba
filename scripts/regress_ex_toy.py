import shutil

from arba.effect import get_effect_list, get_sens_spec
from arba.permute import PermuteRegressVBA

# detection params
par_flag = True
num_perm = 24
alpha = .05

# regression effect params
shape = 5, 5, 5
r2 = .1
effect_num_vox = 10

feat_sbj, file_tree, eff_list = get_effect_list(
    effect_num_vox=effect_num_vox,
    shape=shape, r2=r2,
    rand_seed=1)

eff = eff_list[0]

with file_tree.loaded(effect_list=[eff]):
    perm_reg = PermuteRegressVBA(feat_sbj, file_tree,
                                 num_perm=num_perm,
                                 par_flag=par_flag,
                                 alpha=alpha,
                                 mask_target=eff.mask,
                                 verbose=True)
    estimate_dict = {'arba': perm_reg.mask_estimate}
    estimate_dict.update(perm_reg.vba_mask_estimate_dict)

for label, estimate in estimate_dict.items():
    sens, spec = get_sens_spec(target=eff.mask, estimate=estimate,
                               mask=file_tree.mask)
    print(f'{label}: sens {sens:.3f} spec {spec:.3f}')

shutil.copy(__file__, perm_reg.folder / 'regress_ex_toy.py')
print(perm_reg.folder)
