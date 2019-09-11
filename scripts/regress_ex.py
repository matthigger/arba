import pathlib
import random
import shutil
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.morphology import binary_erosion

from arba.data import SynthFileTree
from arba.effect import get_sens_spec, EffectRegress
from arba.permute import PermuteRegressVBA
from arba.plot import save_fig
from arba.region import RegionRegress
from arba.space import sample_mask_min_var, sample_mask, PointCloud
from pnl_data.set import hcp_100


def get_effect_list(effect_num_vox, file_tree, num_eff=1, min_var_mask=False):
    np.random.seed(1)
    random.seed(1)
    assert file_tree.is_loaded, 'file tree must be loaded'
    prior_array = file_tree.mask

    # erode prior array if edges of prior_array are invalid (TFCE error)
    prior_array = binary_erosion(prior_array)

    # sample effect extent
    eff_mask_list = list()
    for idx in range(num_eff):
        if min_var_mask:
            mask = sample_mask_min_var(num_vox=effect_num_vox,
                                       file_tree=file_tree,
                                       prior_array=prior_array)
        else:
            mask = sample_mask(prior_array=prior_array,
                               num_vox=effect_num_vox,
                               ref=file_tree.ref)
        eff_mask_list.append(mask)
    return eff_mask_list


def plot(folder, method_r2ss_list_dict):
    sns.set(font_scale=1)
    fig, ax = plt.subplots(1, 2)
    for idx, feat in enumerate(('sensitivity', 'specificity')):
        plt.sca(ax[idx])
        for method, r2ss_list in method_r2ss_list_dict.items():
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
    save_fig(folder / 'r2_vs_sens_spec.pdf', size_inches=(10, 4))


if __name__ == '__main__':
    # detection params
    par_flag = True
    num_perm = 24
    alpha = .05

    # regression effect params
    r2_vec = np.logspace(-2, -.5, 2)
    # r2_vec = [.9]
    num_eff = 2
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
        file_tree = SynthFileTree(num_sbj=num_sbj, shape=shape,
                                  mu=np.zeros(dim_img),
                                  cov=np.eye(dim_img),
                                  folder=folder / 'raw_data')
    elif str_img_data == 'hcp100':
        low_res = True,
        feat_tuple = 'fa',
        file_tree = hcp_100.get_file_tree(lim_sbj=num_sbj,
                                          low_res=low_res,
                                          feat_tuple=feat_tuple)

    sbj_feat = np.random.normal(size=(num_sbj, dim_sbj))
    RegionRegress.set_feat_sbj(feat_sbj=sbj_feat, sbj_list=file_tree.sbj_list,
                               append_ones=True)

    eff_list = list()
    with file_tree.loaded():
        # sample effect locations
        eff_mask_list = get_effect_list(effect_num_vox, file_tree,
                                        num_eff=num_eff,
                                        min_var_mask=min_var_effect_locations)

        for eff_mask in eff_mask_list:
            # get img_feat_init, the img_feat before effect is applied
            pc_eff = PointCloud.from_mask(eff_mask)
            fs_dict = RegionRegress.get_fs_dict(file_tree, pc_ijk=pc_eff)
            img_dim = next(iter(fs_dict.values())).d
            feat_img_init = np.empty(shape=(len(fs_dict), img_dim))
            for idx, sbj in enumerate(file_tree.sbj_list):
                feat_img_init[idx, :] = fs_dict[sbj].mu
            img_pool_cov = sum(fs.cov for fs in fs_dict.values()) / len(
                fs_dict)

            # store
            eff_list.append((eff_mask, feat_img_init, img_pool_cov))

        # print
        file_tree.to_nii(folder, mean_flag=True)

        method_r2ss_list_dict = defaultdict(list)
        for eff_idx, (eff_mask, feat_img_init, img_pool_cov) in enumerate(
                eff_list):

            for r2 in r2_vec:
                # build effect
                eff = EffectRegress.from_r2(r2=r2,
                                            img_feat=feat_img_init,
                                            sbj_feat=sbj_feat,
                                            img_pool_cov=img_pool_cov,
                                            mask=eff_mask)
                file_tree.mask = eff.mask.dilate(mask_radius)

                # impose effect on data
                offset = eff.get_offset_array(sbj_feat)
                file_tree.reset_offset(offset)

                # find effects
                _folder = folder / f'eff{eff_idx}_r2_{r2:.2e}'
                perm_reg = PermuteRegressVBA(sbj_feat, file_tree,
                                             num_perm=num_perm,
                                             par_flag=par_flag,
                                             alpha=alpha,
                                             mask_target=eff.mask,
                                             verbose=True,
                                             save_flag=True,
                                             folder=_folder)

                # compute spec / sens
                estimate_dict = {'arba': perm_reg.mask_estimate}
                estimate_dict.update(perm_reg.vba_mask_estimate_dict)

                for method, estimate in estimate_dict.items():
                    sens, spec = get_sens_spec(target=eff.mask,
                                               estimate=estimate,
                                               mask=file_tree.mask)
                    s = f'{method} (r2: {r2:.2e}): sens {sens:.3f} spec {spec:.3f}'
                    print(s)
                    method_r2ss_list_dict[method].append((r2, sens, spec))

    plot(folder, method_r2ss_list_dict)
    print(folder)
