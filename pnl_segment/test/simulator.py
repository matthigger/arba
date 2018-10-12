from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment import simulate
from pnl_segment.adaptive.part_graph_factory import max_kl

n_sim = 2
effect_snr = 10
split_ratio = .5
mask_rad = 1

# build f_img_tree
folder_data = folder / 'dti_in_01193'
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
                                          radius=3,
                                          n=1)
effect = simulate.Effect.from_data(f_img_tree=f_img_tree,
                                   mask=effect_mask,
                                   effect_snr=effect_snr)

# build mask around affected area and neighboring voxels (computational ease)
mask_active = binary_dilation(effect.mask.x, iterations=mask_rad)
mask_active = simulate.Mask(mask_active,
                            ref_space=effect.mask.ref_space)
f_mask_active = mask_active.to_nii(f_ref=f_fa_list[0])

# simulate effect
res_list = sim.run_effect(effect=effect,
                          n=n_sim,
                          f_mask=f_mask_active,
                          part_graph_factory=max_kl,
                          verbose=True)

# # simulate algorithm on healthy population
# f_seg_nii = sim.run_healthy(part_graph_factory=max_kl, verbose=True)
#
