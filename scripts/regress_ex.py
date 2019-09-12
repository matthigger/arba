import pathlib
import random
import shutil
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.morphology import binary_erosion

import arba
from pnl_data.set import hcp_100


def sample_masks(effect_num_vox, data_img, num_eff=1, min_var_mask=False):
    """ gets list of masks (extent of effects) """

    assert data_img.is_loaded, 'file tree must be loaded'
    prior_array = data_img.mask

    # erode prior array if edges of prior_array are invalid (TFCE error)
    prior_array = binary_erosion(prior_array)

    # sample effect extent
    mask_list = list()
    for idx in range(num_eff):
        if min_var_mask:
            mask = arba.space.sample_mask_min_var(num_vox=effect_num_vox,
                                                  data_img=data_img,
                                                  prior_array=prior_array)
        else:
            mask = arba.space.sample_mask(prior_array=prior_array,
                                          num_vox=effect_num_vox,
                                          ref=data_img.ref)
        mask_list.append(mask)
    return mask_list


def sample_effects(r2_vec, data_sbj, **kwargs):
    idx_r2_eff_dict = dict()
    for eff_idx, eff_mask in enumerate(sample_masks(**kwargs)):
        # get feat_img_init, the img_feat before effect is applied
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
                                                    img_feat=feat_img_init,
                                                    feat_sbj=data_sbj.feat,
                                                    img_pool_cov=img_pool_cov,
                                                    mask=eff_mask)

            idx_r2_eff_dict[eff_idx, r2] = eff

    return idx_r2_eff_dict


class Performance:
    """ throwaway: tracks sensitivity and specificty per method """

    def __init__(self):
        self.method_r2ss_list_dict = defaultdict(list)

    def check_in(self, perm_reg):
        # compute spec / sens
        estimate_dict = {'arba': perm_reg.mask_estimate}
        estimate_dict.update(perm_reg.vba_mask_estimate_dict)

        for method, estimate in estimate_dict.items():
            sens, spec = arba.effect.get_sens_spec(target=eff.mask,
                                                   estimate=estimate,
                                                   mask=data_img.mask)
            s = f'{method} (r2: {r2:.2e}): sens {sens:.3f} spec {spec:.3f}'
            print(s)
            self.method_r2ss_list_dict[method].append((r2, sens, spec))

    def plot(self, folder):
        sns.set(font_scale=1)
        fig, ax = plt.subplots(1, 2)
        for idx, feat in enumerate(('sensitivity', 'specificity')):
            plt.sca(ax[idx])
            for method, r2ss_list in self.method_r2ss_list_dict.items():
                r2 = [x[0] for x in r2ss_list]
                vals = [x[1 + idx] for x in r2ss_list]
                plt.scatter(r2, vals, label=method, alpha=.4)

                d = defaultdict(list)
                for _r2, val in zip(r2, vals):
                    d[_r2].append(val)

                x = []
                y = []
                for _r2, val_list in sorted(d.items()):
                    x.append(_r2)
                    y.append(np.mean(val_list))
                plt.plot(x, y)

            plt.ylabel(feat)
            plt.xlabel(r'$r^2$')
            plt.legend()
            plt.gca().set_xscale('log')
        arba.plot.save_fig(folder / 'r2_vs_sens_spec.pdf', size_inches=(10, 4))

        print(folder)


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    # detection params
    par_flag = True
    num_perm = 24
    alpha = .05

    # regression effect params
    # r2_vec =np.logspace(-2, -.5, 2)
    r2_vec = [.9]
    num_eff = 1
    dim_sbj = 1
    num_sbj = 100
    min_var_effect_locations = True

    str_img_data = 'synth'  # 'hcp100' or 'synth'

    mask_radius = 6

    effect_num_vox = 10

    # build dummy folder
    folder = pathlib.Path(tempfile.mkdtemp())
    shutil.copy(__file__, folder / 'regress_ex.py')
    print(folder)

    # duild bummy images
    if str_img_data == 'synth':
        dim_img = 1
        shape = 6, 6, 6
        data_img = arba.data.DataImageSynth(num_sbj=num_sbj, shape=shape,
                                             mu=np.zeros(dim_img),
                                             cov=np.eye(dim_img),
                                             folder=folder / 'raw_data')
    elif str_img_data == 'hcp100':
        low_res = True,
        feat_tuple = 'fa',
        data_img = hcp_100.get_data_img(lim_sbj=num_sbj,
                                          low_res=low_res,
                                          feat_tuple=feat_tuple)

    # build + set feat
    sbj_feat = np.random.normal(size=(num_sbj, dim_sbj))
    data_sbj = arba.data.DataSubject(feat=sbj_feat,
                                         sbj_list=data_img.sbj_list)

    perf = Performance()
    with data_img.loaded():
        data_img.to_nii(folder, mean_flag=True)

        # sample effects
        idx_r2_eff_dict = sample_effects(r2_vec=r2_vec,
                                         data_sbj=data_sbj,
                                         effect_num_vox=effect_num_vox,
                                         data_img=data_img,
                                         num_eff=num_eff,
                                         min_var_mask=min_var_effect_locations)

        # run each effect
        for (eff_idx, r2), eff in idx_r2_eff_dict.items():
            data_img.mask = eff.mask.dilate(mask_radius)

            # impose effect on data
            offset = eff.get_offset_array(data_sbj.feat)
            data_img.reset_offset(offset)

            # find extent
            _folder = folder / f'eff{eff_idx}_r2_{r2:.2e}'
            perm_reg = arba.permute.PermuteRegressVBA(data_sbj, data_img,
                                                      num_perm=num_perm,
                                                      par_flag=par_flag,
                                                      alpha=alpha,
                                                      mask_target=eff.mask,
                                                      verbose=True,
                                                      save_flag=True,
                                                      folder=_folder)

            # record performance
            perf.check_in(perm_reg)
        perf.plot(folder)
