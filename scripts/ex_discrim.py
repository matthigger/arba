import pathlib
import random
from string import ascii_uppercase

import matplotlib
import numpy as np

import arba
from pnl_data.set import hcp_100
from scripts.performance import Performance

matplotlib.use('TkAgg')


def sample_effects(t2_vec, grp_sbj_list_dict, grp_target=None, **kwargs):
    idx_r2_eff_dict = dict()
    for eff_idx, eff_mask in enumerate(arba.space.sample_masks(**kwargs)):
        for t2 in t2_vec:
            eff = arba.effect.EffectConstant.from_t2(t2=t2,
                                                     mask=eff_mask,
                                                     data_img=data_img,
                                                     split=grp_sbj_list_dict,
                                                     grp_target=grp_target)

            idx_r2_eff_dict[eff_idx, t2] = eff

    return idx_r2_eff_dict


if __name__ == '__main__':
    import tempfile
    import shutil

    np.random.seed(1)
    random.seed(1)

    # detection params
    par_flag = True
    num_perm = 24
    alpha = .05

    # regression effect params
    t2_vec = np.logspace(-1, .5, 4)
    # t2_vec = [.1]
    num_eff = 1
    num_sbj = 100
    min_var_effect_locations = False

    str_img_data = 'hcp100'  # 'hcp100' or 'synth'

    mask_radius = 5

    effect_num_vox = 200

    # build dummy folder
    folder = pathlib.Path(tempfile.mkdtemp())
    print(folder)
    shutil.copy(__file__, folder / pathlib.Path(__file__).name)

    # duild bummy images
    if str_img_data == 'synth':
        dim_img = 1
        shape = 10, 10, 10
        data = np.random.standard_normal((*shape, num_sbj, dim_img))
        data_img = arba.data.DataImageArray(data)
        data_img.to_nii(folder=folder / 'raw_data')

    elif str_img_data == 'hcp100':
        low_res = True,
        feat_tuple = 'fa',
        data_img = hcp_100.get_data_image(lim_sbj=num_sbj,
                                          low_res=low_res,
                                          feat_tuple=feat_tuple)

    # split data into two even groups
    half = int(len(data_img.sbj_list) / 2)
    split = arba.permute.Split({'grp0': data_img.sbj_list[: half],
                                'grp1': data_img.sbj_list[half:]})
    arba.region.RegionDiscriminate.set_split(split)

    t2_letter_dict = dict(zip(t2_vec, ascii_uppercase))

    perf = Performance(stat_label='t2')
    with data_img.loaded():
        mask_img_full = data_img.mask
        data_img.to_nii(folder, mean=True)

        # sample effects
        idx_t2_eff_dict = sample_effects(t2_vec=t2_vec,
                                         data_img=data_img,
                                         grp_sbj_list_dict=split,
                                         effect_num_vox=effect_num_vox,
                                         num_eff=num_eff,
                                         min_var_mask=min_var_effect_locations,
                                         grp_target='grp1')

        # run each effect
        for (eff_idx, t2), eff in idx_t2_eff_dict.items():
            # build smaller DataImage which cuts the zeros out
            _data_img, \
            trim_slice = data_img.trim(mask=eff.mask.dilate(mask_radius),
                                       n_buff=1)
            eff.mask = eff.mask[trim_slice]
            eff.mask.ref = _data_img.ref

            # impose effect on data
            sbj_bool = split.get_sbj_bool(_data_img.sbj_list)
            offset = eff.get_offset(shape=_data_img.ref.shape,
                                    sbj_bool=sbj_bool)
            _data_img.reset_offset(offset)

            # estimate extents
            _folder = folder / f'eff{eff_idx}_t2_{t2_letter_dict[t2]}_{t2:.2e}'
            perm_reg = arba.permute.PermuteDiscriminateVBA(_data_img,
                                                           split=split,
                                                           num_perm=num_perm,
                                                           par_flag=par_flag,
                                                           alpha=alpha,
                                                           mask_target=eff.mask,
                                                           verbose=True,
                                                           folder=_folder)

            perm_reg.save(size_v_saf=True, null=True, size_v_f=True)

            # record performance
            perf.check_in(stat=t2, perm_reg=perm_reg)

            print(_folder)
        perf.plot(folder)
