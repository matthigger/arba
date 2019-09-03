import pathlib
import shutil
import tempfile

import arba
from arba.effect.effect_regress.sample import get_effect_list

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
        arba.permute.run_permute(feat_sbj, file_tree,
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

mask_estimate = merge_record.build_mask(sig_node_cover)
mask_estimate.to_nii(folder / 'mask_estimate.nii.gz')
arba.permute.compute_print_dice(mask_estimate=mask_estimate,
                                mask_target=eff.mask,
                                save_folder=folder)

print(folder)
