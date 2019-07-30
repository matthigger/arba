import pathlib
import shutil
import tempfile

import numpy as np

import arba

dim_sbj = 1
dim_img = 1

# subject params
mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 50

# imaging params
mu_img = np.zeros(dim_img)
sig_img = np.eye(dim_img)
shape = 6, 6, 6

# detection params
num_perm = 3

# regression params
r2 = .5
mask = np.zeros(shape)
mask[2:5, 2:5, 2:5] = True

# build effect
ref = arba.space.RefSpace(affine=np.eye(4))
mask = arba.space.Mask(mask, ref=ref)

# build output folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'ex.py')

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
    fs = file_tree.get_fs(mask=mask)
eff = arba.simulate.EffectRegress.from_r2(r2=r2,
                                          mask=mask,
                                          eps_img=fs.cov,
                                          cov_sbj=np.cov(feat_sbj.T, ddof=0),
                                          feat_mapper=feat_mapper)


#
def mse(reg, **kwargs):
    return reg.mse


def weighted_r2(reg, **kwargs):
    return reg.r2 * len(reg)


f_mask = folder / 'target_mask.nii.gz'
mask.to_nii(f_mask)

fnc_tuple = mse, weighted_r2
with file_tree.loaded(effect_list=[eff]):
    sg_hist, reg_list, val_list = \
        arba.regress.run_permute(feat_sbj, file_tree,
                                 fnc_target=weighted_r2,
                                 save_folder=folder,
                                 max_flag=True,
                                 cutoff_perc=95,
                                 n=num_perm,
                                 fnc_tuple=fnc_tuple)

node_mask, d_max = sg_hist.merge_record.get_node_max_dice(mask)

sg_hist.merge_record.plot_size_v(weighted_r2, label='n * r2', mask=mask,
                                 log_y=True)
arba.plot.save_fig(folder / 'size_v_nr2.pdf')

sg_hist.merge_record.plot_size_v(mse, label='mse', mask=mask)
arba.plot.save_fig(folder / 'size_v_mse.pdf')

arba.regress.compute_print_dice(reg_list, mask_target=mask, save_folder=folder)
print(folder)
