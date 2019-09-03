import pathlib
import random
import shutil
import tempfile

import numpy as np

import arba


def get_effect_list(effect_num_vox, shape, num_eff=1, r2=.5, dim_sbj=1,
                    dim_img=1, num_sbj=100, rand_seed=None):
    if rand_seed is not None:
        np.random.seed(1)
        random.seed(1)

    # subject params
    mu_sbj = np.zeros(dim_sbj)
    sig_sbj = np.eye(dim_sbj)

    # imaging params
    mu_img = np.zeros(dim_img)
    sig_img = np.eye(dim_img)

    # dummy reference space
    ref = arba.space.RefSpace(affine=np.eye(4))

    # sample sbj features
    feat_sbj = np.random.multivariate_normal(mean=mu_sbj,
                                             cov=sig_sbj,
                                             size=num_sbj)

    # build feat_img (shape0, shape1, shape2, num_sbj, dim_img)
    feat_img = np.random.multivariate_normal(mean=mu_img,
                                             cov=sig_img,
                                             size=(*shape, num_sbj))

    # build file_tree
    file_tree = arba.data.SynthFileTree.from_array(data=feat_img,
                                                   folder=folder / 'data')

    feat_mapper = arba.regress.FeatMapperStatic(n=dim_sbj,
                                                sbj_list=file_tree.sbj_list,
                                                feat_sbj=feat_sbj)

    # build regression, impose it
    eff_list = list()
    prior_array = np.ones(shape)
    cov_sbj = np.cov(feat_sbj.T, ddof=0)
    with file_tree.loaded():
        for idx in range(num_eff):
            # sample effect extent
            effect_mask = arba.space.sample_mask(prior_array=prior_array,
                                                 num_vox=effect_num_vox,
                                                 ref=ref)

            # get imaging feature stats in this mask
            fs = file_tree.get_fs(mask=effect_mask)

            # construct effect
            eff = arba.simulate.EffectRegress.from_r2(r2=r2,
                                                      mask=effect_mask,
                                                      eps_img=fs.cov,
                                                      cov_sbj=cov_sbj,
                                                      feat_mapper=feat_mapper)
            eff_list.append(eff)

    return feat_sbj, file_tree, eff_list


if __name__ == '__main__':
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
        sg_hist, node_pval_dict, node_z_dict, r2_null = \
            arba.regress.run_permute(feat_sbj, file_tree,
                                     save_folder=folder,
                                     num_perm=num_perm,
                                     par_flag=par_flag)

        merge_record = sg_hist.merge_record
        node_p_negz_dict = {n: (p, -node_z_dict[n])
                            for n, p in node_pval_dict.items() if p <= alpha}
        sig_node_cover = merge_record._cut_greedy(node_p_negz_dict,
                                                  max_flag=False)
        for n in sig_node_cover:
            r = sg_hist.merge_record.resolve_node(n,
                                                  file_tree=file_tree,
                                                  reg_cls=arba.region.RegionRegress)
            r.pc_ijk.to_mask().to_nii(folder / f'node_{n}.nii.gz')
            r.plot(img_feat_label='fa')
            arba.plot.save_fig(folder / f'node_{n}.pdf')

    # node_mask, d_max = merge_record.get_node_max_dice(effect_mask)

    merge_record.plot_size_v('r2', label='r2', mask=eff.mask, log_y=False)
    arba.plot.save_fig(folder / 'size_v_r2.pdf')

    merge_record.plot_size_v(node_pval_dict, label='pval', mask=eff.mask,
                             log_y=False)
    arba.plot.save_fig(folder / 'size_v_pval.pdf')

    merge_record.plot_size_v(node_z_dict, label='r2z', mask=eff.mask,
                             log_y=False)
    arba.plot.save_fig(folder / 'size_v_r2z_score.pdf')

    mask_estimate = arba.regress.build_mask(sig_node_cover, merge_record)
    mask_estimate.to_nii(folder / 'mask_estimate.nii.gz')
    arba.regress.compute_print_dice(mask_estimate=mask_estimate,
                                    mask_target=eff.mask,
                                    save_folder=folder)

    print(folder)
