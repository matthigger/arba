import shutil

from arba.effect.effect_regress.sample import get_effect_list
from arba.permute import PermuteRegress

# detection params
par_flag = True
num_perm = 24
alpha = .05

# regression effect params
shape = 6, 6, 6
r2 = .5
effect_num_vox = 27

feat_sbj, file_tree, eff_list = get_effect_list(
    effect_num_vox=effect_num_vox,
    shape=shape, r2=r2,
    rand_seed=1)

eff = eff_list[0]

with file_tree.loaded(effect_list=[eff]):
    perm_reg = PermuteRegress(feat_sbj, file_tree,
                              num_perm=num_perm,
                              par_flag=par_flag,
                              alpha=alpha,
                              target_mask=eff.mask,
                              verbose=True)
shutil.copy(__file__, perm_reg.folder / 'regress_ex_toy.py')
print(perm_reg.folder)
