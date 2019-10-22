import pathlib
import random
import shutil

import numpy as np

import arba
from arba.space.mask.sample import sample_masks
from mh_pytools import file
from pnl_data.set import hcp_100
from scripts.performance import Performance


def sample_effects(r2_vec, data_sbj, **kwargs):
    idx_r2_eff_dict = dict()
    for eff_idx, eff_mask in enumerate(sample_masks(**kwargs)):
        # get feat_img_init, the feat_img before effect is applied
        pc_eff = arba.space.PointCloud.from_mask(eff_mask)
        fs_dict = arba.region.RegionRegress.get_fs_dict(data_img,
                                                        pc_ijk=pc_eff)
        img_dim = next(iter(fs_dict.values())).d

        feat_img_init = np.empty(shape=(len(fs_dict), img_dim))
        for idx, sbj in enumerate(data_img.sbj_list):
            feat_img_init[idx, :] = fs_dict[sbj].mu
        img_pool_cov = sum(fs.cov for fs in fs_dict.values()) / len(fs_dict)

        for r2 in r2_vec:
            eff = arba.effect.EffectRegress.from_r2(r2=r2,
                                                    feat_img=feat_img_init,
                                                    feat_sbj=data_sbj.feat,
                                                    contrast=data_sbj.contrast,
                                                    img_pool_cov=img_pool_cov,
                                                    mask=eff_mask)

            idx_r2_eff_dict[eff_idx, r2] = eff

    return idx_r2_eff_dict


if __name__ == '__main__':
    import tempfile

    np.random.seed(1)
    random.seed(1)

    # detection params
    par_flag = True
    num_perm = 24
    alpha = .05

    # regression effect params
    # r2_vec = np.logspace(-2, -.1, 7)
    r2_vec = [.2]
    num_eff = 1
    num_sbj = 20
    min_var_effect_locations = False

    str_img_data = 'hcp100'  # 'hcp100' or 'synth'

    mask_radius = 1000

    effect_num_vox = 150

    # build dummy folder
    folder = pathlib.Path(tempfile.mkdtemp())
    # folder = pathlib.Path('/home/mu584/dropbox_pnl/arba_reg_ex/hcp_age_sex')
    # assert not folder.exists(), 'folder exists'
    # folder.mkdir(parents=True)
    shutil.copy(__file__, folder / pathlib.Path(__file__).name)
    print(folder)

    # duild bummy images
    if str_img_data == 'synth':
        contrast = [1]
        dim_sbj = 1
        dim_img = 1
        shape = 10, 10, 10
        data = np.random.standard_normal((*shape, num_sbj, dim_img))
        data_img = arba.data.DataImageArray(data)
        data_img.to_nii(folder=folder / 'raw_data')

        sbj_feat = np.random.normal(size=(num_sbj, dim_sbj))
        data_sbj = arba.data.DataSubject(feat=sbj_feat,
                                         sbj_list=data_img.sbj_list,
                                         contrast=contrast)

    elif str_img_data == 'hcp100':
        low_res = True,
        feat_tuple = 'fa',
        contrast = [1, 1]
        data_img = hcp_100.get_data_image(lim_sbj=num_sbj,
                                          low_res=low_res,
                                          feat_tuple=feat_tuple)

        age_sex = hcp_100.age_sex_array(data_img.sbj_list)
        data_sbj = arba.data.DataSubject(feat=age_sex,
                                         feat_list=['age_grp', 'sex'],
                                         sbj_list=data_img.sbj_list,
                                         contrast=contrast)

    perf = Performance(stat_label='r2')
    with data_img.loaded():
        mask_img_full = data_img.mask
        data_img.to_nii(folder, mean=True)

        # sample effects
        idx_r2_eff_dict = sample_effects(r2_vec=r2_vec,
                                         data_sbj=data_sbj,
                                         effect_num_vox=effect_num_vox,
                                         data_img=data_img,
                                         num_eff=num_eff,
                                         min_var_mask=min_var_effect_locations)

        # run each effect
        for (eff_idx, r2), eff in idx_r2_eff_dict.items():
            # build smaller DataImage which cuts the zeros out
            _data_img, \
            trim_slice = data_img.trim(mask=eff.mask.dilate(mask_radius),
                                       n_buff=1)
            eff.mask = eff.mask[trim_slice]
            eff.mask.ref = _data_img.ref

            # impose effect on data
            offset = eff.get_offset_array(data_sbj.feat)
            _data_img.reset_offset(offset)

            # find extent
            _folder = folder / f'eff{eff_idx}_r2_{r2:.2e}'
            file.save(eff, _folder / 'effect.p.gz')
            perm_reg = arba.permute.PermuteRegressVBA(data_img=_data_img,
                                                      data_sbj=data_sbj,
                                                      num_perm=num_perm,
                                                      par_flag=par_flag,
                                                      alpha=alpha,
                                                      mask_target=eff.mask,
                                                      verbose=True,
                                                      folder=_folder)
            perm_reg.save(size_v_z=True, null=True, size_v_stat=True)

            # record performance
            perf.check_in(stat=r2, perm_reg=perm_reg)
        perf.plot(folder)
