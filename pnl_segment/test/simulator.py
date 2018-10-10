from collections import defaultdict
from functools import reduce

import nibabel as nib
import numpy as np

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment.simulate.effect import Effect
from pnl_segment.simulate.simulator import Simulator

effect_snr = 1

# build f_img_tree
folder_data = folder / 'dti_in_01193'
f_img_tree = defaultdict(dict)
for label in ('fa', 'md'):
    for f in folder_data.glob(f'*{label}.nii.gz'):
        f_img_tree[get_name(f.stem)][label] = f

# sample a mask
np.random.seed(1)
f_fa_iter = iter(nib.load(str(f_img_tree[sbj]['fa'])).get_data()
                 for sbj in f_img_tree.keys())
eff_center_prior = reduce(np.logical_and, f_fa_iter).astype(int)
mask = Effect.sample_mask(prior_array=eff_center_prior, radius=4, n=1)

# output mask to nii
f_fa = next(iter(f_img_tree.values()))['fa']
f_mask = mask.to_nii(f_ref=f_fa)

# build effect
effect = Effect.from_data(f_img_tree=f_img_tree,
                          mask=mask,
                          effect_snr=effect_snr)

# simulate effect on population
s = Simulator(f_img_tree, effect=effect)
f_img_tree_eff = s.sample_all_eff()
