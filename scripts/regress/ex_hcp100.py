import pathlib
import random
import shutil
import tempfile

import numpy as np
from scipy.ndimage import binary_dilation

import arba
from pnl_data.set import hcp_100

# fix random seed
np.random.seed(1)
random.seed(1)

# subject params
dim_sbj = 1
mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)

# detection params
num_perm = 0

file_tree = hcp_100.get_file_tree(low_res=True, feat_tuple=('fa',))

# regression params
r2 = .99
effect_size = 100
mask_rad = 3
with file_tree.loaded():
    effect_mask = arba.space.sample_mask_min_var(file_tree=file_tree,
                                                 num_vox=effect_size)
    fs = file_tree.get_fs(mask=effect_mask)
file_tree.mask = binary_dilation(effect_mask, iterations=mask_rad)

# build output folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'ex_toy.py')

# sample sbj features
feat_sbj = np.random.multivariate_normal(mean=mu_sbj,
                                         cov=sig_sbj,
                                         size=file_tree.num_sbj)

feat_mapper = arba.regress.FeatMapperStatic(n=dim_sbj,
                                            sbj_list=file_tree.sbj_list,
                                            feat_sbj=feat_sbj)

# build regression effect
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


f_mask = folder / 'target_mask.nii.gz'
effect_mask.to_nii(f_mask)

fnc_tuple = mse, maha_zero, r2
with file_tree.loaded(effect_list=[eff]):
    sg_hist, sig_node_list, val_list = \
        arba.regress.run_permute(feat_sbj, file_tree,
                                 fnc_target=maha_zero,
                                 save_folder=folder,
                                 max_flag=True,
                                 cutoff_perc=95,
                                 n=num_perm,
                                 fnc_tuple=fnc_tuple)

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
