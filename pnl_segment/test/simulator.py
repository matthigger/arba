import random
from collections import defaultdict

import numpy as np

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment import simulate
from pnl_segment.adaptive.part_graph_factory import max_kl

effect_snr = 10
split_ratio = .5

# get the same random each time for ...
np.random.seed(1)  # ... building mask
random.seed(1)  # ... split healthy from effects

# build f_img_tree
folder_data = folder / 'dti_in_01193'
f_img_tree = defaultdict(dict)
for label in ('fa', 'md'):
    for f in folder_data.glob(f'*{label}.nii.gz'):
        f_img_tree[get_name(f.stem)][label] = f

# build mask of intersection of all fa values
f_fa_list = [f_img_tree[sbj]['fa'] for sbj in f_img_tree.keys()]
mask_all = simulate.Mask.build_intersection_from_nii(f_fa_list)

# sample a mask
mask = simulate.Effect.sample_mask(prior_array=mask_all.x, radius=4, n=1)

# output mask to nii
f_fa = next(iter(f_img_tree.values()))['fa']
f_mask = mask.to_nii(f_ref=f_fa)

# build effect
effect = simulate.Effect.from_data(f_img_tree=f_img_tree,
                                   mask=mask,
                                   effect_snr=effect_snr)

# simulate effect on population
s = simulate.Simulator(f_img_health=f_img_tree, split_ratio=split_ratio)
f_seg_nii = s.run_healthy(part_graph_factory=max_kl, verbose=True)
