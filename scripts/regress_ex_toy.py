import pathlib
import shutil
import tempfile

import arba
from arba.effect.effect_regress.sample import get_effect_list

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

# build output folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'regress_ex_toy.py')

eff = eff_list[0]

f_mask = folder / 'target_mask.nii.gz'
eff.mask.to_nii(f_mask)

with file_tree.loaded(effect_list=[eff]):
    sg_hist, node_pval_dict, node_z_dict, r2_null, sig_node_cover = \
        arba.permute.run_permute(feat_sbj, file_tree,
                                 save_folder=folder,
                                 num_perm=num_perm,
                                 par_flag=par_flag,
                                 alpha=alpha,
                                 target_mask=eff.mask)

print(folder)
