import pathlib
import random
import shutil
import tempfile

import numpy as np

import arba

# fix random seed
np.random.seed(1)
random.seed(1)

dim_sbj = 1
dim_img = 1

reg_size_thresh = 1

# subject params
mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 100

# imaging params
mu_img = np.zeros(dim_img)
sig_img = np.eye(dim_img)
shape = 6, 6, 6

# detection params
par_flag = True
num_perm = 320

# regression params
r2 = .5
effect_mask = np.zeros(shape)
effect_mask[2:5, 2:5, 2:5] = True

# build effect
ref = arba.space.RefSpace(affine=np.eye(4))
effect_mask = arba.space.Mask(effect_mask, ref=ref)

# build output folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'ex_toy.py')

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
with file_tree.loaded():
    fs = file_tree.get_fs(mask=effect_mask)
eff = arba.simulate.EffectRegress.from_r2(r2=r2,
                                          mask=effect_mask,
                                          eps_img=fs.cov,
                                          cov_sbj=np.cov(feat_sbj.T, ddof=0),
                                          feat_mapper=feat_mapper)


#
def mse(reg, **kwargs):
    return reg.mse


def r2(reg, **kwargs):
    return reg.r2


def maha_zero(reg, **kwargs):
    return reg.maha[0]

min
f_mask = folder / 'target_mask.nii.gz'
effect_mask.to_nii(f_mask)

fnc_tuple = mse, maha_zero, r2
with file_tree.loaded(effect_list=[eff]):
    sg_hist, sig_node_list, val_list = \
        arba.regress.run_permute(feat_sbj, file_tree,
                                 fnc_target=r2,
                                 save_folder=folder,
                                 max_flag=True,
                                 cutoff_perc=95,
                                 n=num_perm,
                                 fnc_tuple=fnc_tuple,
                                 reg_size_thresh=reg_size_thresh,
                                 par_flag=par_flag)

    sig_node_cover = sg_hist.merge_record.get_cover(sig_node_list)
    for n in sig_node_cover:
        r = sg_hist.merge_record.resolve_node(n,
                                              file_tree=file_tree,
                                              reg_cls=arba.region.RegionRegress)
        r.pc_ijk.to_mask().to_nii(folder / f'node_{n}.nii.gz')
        r.plot(img_feat_label='fa')
        arba.plot.save_fig(folder / f'node_{n}.pdf')

node_mask, d_max = sg_hist.merge_record.get_node_max_dice(effect_mask)

sg_hist.merge_record.plot_size_v(maha_zero, label='maha(0)', mask=effect_mask,
                                 log_y=True)
arba.plot.save_fig(folder / 'size_v_maha0.pdf')

sg_hist.merge_record.plot_size_v(r2, label='r2', mask=effect_mask,
                                 log_y=False)
arba.plot.save_fig(folder / 'size_v_r2.pdf')

sg_hist.merge_record.plot_size_v(mse, label='mse', mask=effect_mask)
arba.plot.save_fig(folder / 'size_v_mse.pdf')

mask_estimate = arba.regress.build_mask(sig_node_list, sg_hist.merge_record)
mask_estimate.to_nii(folder / 'mask_estimate.nii.gz')
arba.regress.compute_print_dice(mask_estimate=mask_estimate,
                                mask_target=effect_mask, save_folder=folder)
print(folder)
