import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from mh_pytools import parallel, file
from pnl_data.set.cidar_post import folder, get_name
from pnl_segment import simulate

n_effect = 1000
effect_snr_iter = np.geomspace(.001, 5, n_effect)
eff_ratio = .5
effect_rad = 3
mask_rad = 5
obj = 'max_maha'
prev_folder_out = folder / '2018_Oct_17_09_15AM24'

folder_data = folder / 'dti_in_01193'
if prev_folder_out is None:
    folder_out = folder / datetime.now().strftime("%Y_%b_%d_%I_%M%p%S")
    folder_out.mkdir(exist_ok=True, parents=True)

    # build f_img_health
    f_img_health = defaultdict(dict)
    for label in ('fa', 'md'):
        for f in folder_data.glob(f'*{label}.nii.gz'):
            f_img_health[get_name(f.stem)][label] = f

    # init simulator, split into groups
    sim = simulate.Simulator(f_img_health=f_img_health)
    sbj_effect, sbj_health = sim.split_sbj(eff_ratio=.5)
    file.save((sim, sbj_effect, sbj_health),
              file=folder_out / 'sim_split.p.gz')

    # single healthy (serial) run
    folder_healthy = folder_out / '_no_effect'
    sim.run_healthy(obj=obj, save=True, verbose=True, folder=folder_healthy)
else:
    # allows us to add to previous experiments
    folder_out = prev_folder_out
    sim, sbj_effect, sbj_health = file.load(folder_out / 'sim_split.p.gz')

# build intersection of all fa as mask of 'prior' location of effect
f_fa_list = [sim.f_img_health[sbj]['fa'] for sbj in sim.f_img_health.keys()]
mask_all = simulate.Mask.build_intersection_from_nii(f_fa_list)


# build effect
def get_effect(snr):
    """ generates effect at random location with given snr to healthy
    """
    effect_mask = simulate.Effect.sample_mask(prior_array=mask_all.x,
                                              radius=effect_rad)
    effect = simulate.Effect.from_data(f_img_tree=sim.f_img_health,
                                       mask=effect_mask,
                                       effect_snr=snr)

    # build mask around affected area and neighboring voxels (faster compute)
    mask_active = binary_dilation(effect.mask.x, iterations=mask_rad)
    mask_active = simulate.Mask(mask_active,
                                ref_space=effect.mask.ref_space)
    f_mask_active = mask_active.to_nii(f_ref=f_fa_list[0])

    return effect, f_mask_active


# build list of args into run_effect, each its own effect location / snr
arg_list = list()
for snr in tqdm(effect_snr_iter, desc='sample effects'):
    effect, f_mask_active = get_effect(snr)
    d = {'obj': obj,
         'save': True,
         'effect': effect,
         'sbj_effect': sbj_effect,
         'sbj_health': sbj_health,
         'f_mask': f_mask_active,
         'folder': folder_out / f'{snr:.3E}_{uuid.uuid4().hex[:4]}'}
    arg_list.append(d)

# run
parallel.run_par_fnc(fnc='run_effect', obj=sim, arg_list=arg_list)
