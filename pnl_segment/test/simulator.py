from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment import simulate

n_healthy = 10
effect_snr = 2
split_ratio = .5
effect_rad = 3
mask_rad = 5
obj = 'max_maha'

# build f_img_tree
folder_data = folder / 'dti_in_01193'
folder_out = folder / 'dti_in_01193_out'
f_img_tree = defaultdict(dict)
for label in ('fa', 'md'):
    for f in folder_data.glob(f'*{label}.nii.gz'):
        f_img_tree[get_name(f.stem)][label] = f

# init simulator
sim = simulate.Simulator(f_img_health=f_img_tree, split_ratio=split_ratio)

# build a consistently random effect (debug)
np.random.seed(1)

# build intersection of all fa as mask of 'prior' location of effect
f_fa_list = [f_img_tree[sbj]['fa'] for sbj in f_img_tree.keys()]
mask_all = simulate.Mask.build_intersection_from_nii(f_fa_list)

# build effect
effect_mask = simulate.Effect.sample_mask(prior_array=mask_all.x,
                                          radius=effect_rad)
effect = simulate.Effect.from_data(f_img_tree=f_img_tree,
                                   mask=effect_mask,
                                   effect_snr=effect_snr)

# build mask around affected area and neighboring voxels (computational ease)
mask_active = binary_dilation(effect.mask.x, iterations=mask_rad)
mask_active = simulate.Mask(mask_active,
                            ref_space=effect.mask.ref_space)
f_mask_active = mask_active.to_nii(f_ref=f_fa_list[0])

# simulate effect
_, folder = sim.run_effect(obj=obj,
                           effect=effect,
                           f_mask=f_mask_active,
                           verbose=True,
                           save=True)

f_stat = folder / 'segment_stat.nii.gz'
auc = effect.get_auc_from_nii(f_stat)

# simulate algorithm on healthy population

# single healthy (serial) run
# _, folder = sim.run_healthy(obj=obj, save=True, verbose=True)

# # n_healthy healthy (parallel) run
# timestamp = datetime.now().strftime("%Y_%b_%d_%I_%M%p%S")
# folder_healthy = folder_out / f'no_eff_full_brain_{timestamp}'
# arg_list = [{'obj': obj, 'save': True,
#              'folder': folder_healthy / f'{obj}_{idx}'} for idx in range(n_healthy)]
# parallel.run_par_fnc(fnc='run_healthy', obj=sim, arg_list=arg_list)
