import pathlib
import shutil
import tempfile

import numpy as np

import arba

dim_sbj = 1
dim_img = 1

mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 100

shape = 8, 8, 8
mask = np.zeros(shape)
mask[2:5, 2:5, 2:5] = True
ref = arba.space.RefSpace(affine=np.eye(4))
mask = arba.space.Mask(mask, ref=ref)

np.random.seed(1)
# beta = np.random.normal(0, 1, (dim_sbj, dim_img))
beta = np.atleast_2d(1)
print(f'beta = {beta}')
cov_eps = np.diag(np.random.normal(0, 1, dim_img) ** 2)

folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'ex.py')

# sample sbj features
feat_sbj = np.random.multivariate_normal(mean=mu_sbj, cov=sig_sbj,
                                         size=num_sbj)

# build feat_img (shape0, shape1, shape2, num_sbj, dim_img)
feat_img = np.random.multivariate_normal(mean=np.zeros(dim_img),
                                         cov=cov_eps,
                                         size=(*shape, num_sbj))
# add offset to induce beta
for sbj_idx, offset in enumerate(feat_sbj @ beta):
    feat_img[mask, sbj_idx, :] += offset

# add offset to introduce second 'tissue type'
feat_img[2:5, 5:, 5:, ...] += 1

# build file_tree
file_tree = arba.data.SynthFileTree.from_array(data=feat_img,
                                               folder=folder / 'data')


#
def mse(reg, **kwargs):
    return reg.mse


def aic(reg, **kwargs):
    return 2 * (reg.beta.size - reg.log_like)


def bic(reg, **kwargs):
    n = len(reg) * len(reg.sbj_list)
    k = reg.beta.size
    return (np.log(n) * k - 2 * reg.log_like) / n


def mse_delta_norm(reg, reg_tuple=None):
    if reg_tuple is None:
        return 0
    else:
        se_kids = sum(mse(r) * len(r) for r in reg_tuple) / len(reg)
    return max(reg.mse - r.mse for r in reg_tuple)


def r2_delta_norm(reg, reg_tuple=None):
    if reg_tuple is None:
        return 0
    else:
        r2_kids = max(r.r2 for r in reg_tuple)
    return reg.r2 - r2_kids


def r2(reg, **kwargs):
    return reg.r2


def log_like_norm(reg, **kwargs):
    return reg.log_like / len(reg)


f_mask = folder / 'target_mask.nii.gz'
mask.to_nii(f_mask)

# set feat_sbj
arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                       sbj_list=file_tree.sbj_list)

with file_tree.loaded():
    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=file_tree,
                                             cls_reg=arba.region.RegionRegress,
                                             fnc_tuple=(
                                                 mse, mse_delta_norm,
                                                 log_like_norm, r2))

    # sg_hist.merge_by_atlas(f_mask, skip_zero=False)
    sg_hist.reduce_to(1, verbose=True)

    merge_record = sg_hist.merge_record
    merge_record.to_nii(folder / 'seg_hier.nii.gz', n=10)

    node_mse_dict = merge_record.fnc_node_val_list[mse_delta_norm]
    node_list = sg_hist._cut_greedy(node_mse_dict, max_flag=False)
    reg_list = [merge_record.resolve_node(node=n,
                                          file_tree=file_tree,
                                          reg_cls=arba.region.RegionRegress)
                for n in node_list]

    node_mask, d_max = sg_hist.merge_record.get_node_max_dice(mask)
    reg_mask = sg_hist.merge_record.resolve_node(node=node_mask,
                                                 file_tree=file_tree,
                                                 reg_cls=arba.region.RegionRegress)

sg_hist.merge_record.plot_size_v(mse, label='mse', mask=mask)
arba.plot.save_fig(folder / 'size_v_mse.pdf')

sg_hist.merge_record.plot_size_v(mse_delta_norm, label='mse_delta_max',
                                 mask=mask)
arba.plot.save_fig(folder / 'size_v_mse_norm.pdf')

sg_hist.merge_record.plot_size_v(r2, label='r2', mask=mask)
arba.plot.save_fig(folder / 'size_v_r2.pdf')

sg_hist.merge_record.plot_size_v(log_like_norm,
                                 label='log likely (normalized)', mask=mask)
arba.plot.save_fig(folder / 'size_v_log_like.pdf')

reg_list = sorted(reg_list, key=lambda r: r.mse, reverse=False)

for idx, r in enumerate(reg_list[:3]):
    r.plot()
    arba.plot.save_fig(f_out=folder / f'reg_{idx}_scatter.pdf')
    r.pc_ijk.to_mask().to_nii(f_out=folder / f'reg_{idx}_mask.nii.gz')
