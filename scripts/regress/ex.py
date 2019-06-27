import pathlib
import tempfile

import numpy as np

import arba

dim_sbj = 1
dim_img = 1

mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 10

shape = 10, 10, 10
mask = np.zeros(shape)
mask[3:7, 3:7, 3:7] = True
ref = arba.space.RefSpace(affine=np.eye(4))
mask = arba.space.Mask(mask, ref=ref)

beta = np.random.normal(0, 10, (dim_sbj, dim_img))
cov_eps = np.diag(np.random.normal(0, 1, dim_img) ** 2)

folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)

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

# build file_tree
file_tree = arba.data.SynthFileTree.from_array(data=feat_img,
                                               folder=folder / 'data')


#
def r2(reg):
    return reg.r2


f_mask = folder / 'target_mask.nii.gz'
mask.to_nii(f_mask)

# set feat_sbj
arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                       sbj_list=file_tree.sbj_list)

with file_tree.loaded():
    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=file_tree,
                                             cls_reg=arba.region.RegionRegress,
                                             fnc_tuple=(r2,))

    # sg_hist.merge_by_atlas(f_mask, skip_zero=False)
    sg_hist.reduce_to(1, verbose=True)

    merge_record = sg_hist.merge_record
    merge_record.to_nii(folder / 'seg_hier.nii.gz', n=10)

    node_r2_dict = merge_record.fnc_node_val_list[r2]
    node_list = sg_hist._cut_greedy(node_r2_dict, max_flag=True)
    reg_list = [merge_record.resolve_node(node=n,
                                          file_tree=file_tree,
                                          reg_cls=arba.region.RegionRegress)
                for n in node_list]

    node_mask, d_max = sg_hist.merge_record.get_node_max_dice(mask)
    reg_mask = sg_hist.merge_record.resolve_node(node=node_mask,
                                                 file_tree=file_tree,
                                                 reg_cls=arba.region.RegionRegress)

sg_hist.merge_record.plot_size_v(r2, label='r2', mask=mask)
arba.plot.save_fig(folder / 'size_v_r2.pdf')

reg_list = sorted(reg_list, key=lambda r: r.r2, reverse=True)

for idx, r in enumerate(reg_list[:10]):
    r.plot()
    arba.plot.save_fig(f_out=folder / f'reg_{idx}_scatter.pdf')
    r.pc_ijk.to_mask().to_nii(f_out=folder / f'reg_{idx}_mask.nii.gz')
