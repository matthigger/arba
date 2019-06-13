import pathlib
import shutil
import tempfile

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage.morphology import binary_dilation

import arba
from mh_pytools import file
from pnl_data.set.hcp_100 import get_file_tree

# file tree
n_sbj = 10
mu = (0, 0)
cov = np.eye(2) * 1
shape = 10, 10, 10

offset = (1, 1)

fake_data = False
one_vs_many = False

np.random.seed(1)

# build file tree
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)
folder.mkdir(exist_ok=True, parents=True)

if fake_data:
    # build file tree
    _folder = folder / 'data_orig'
    _folder.mkdir(exist_ok=True, parents=True)
    ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                                 folder=_folder)

    # effect
    mask = np.zeros(shape)
    mask[0:5, 0:5, 0:5] = 1
    mask = arba.space.Mask(mask, ref=ft.ref)
    effect = arba.simulate.Effect(mask=mask, offset=offset)
else:
    # get file tree
    ft = get_file_tree(lim_sbj=10, low_res=True)

    # build effect
    with ft.loaded():
        eff_mask = arba.space.sample_mask(ft.mask, num_vox=200, ref=ft.ref)
        fs = ft.get_fs(mask=eff_mask)
        effect = arba.simulate.Effect.from_fs_t2(fs=fs, t2=1, mask=eff_mask,
                                                 u=offset)

    # mask file_tree (lowers compute needed)
    mask = binary_dilation(eff_mask, iterations=4)
    ft.mask = np.logical_and(mask, ft.mask)

# copy script to output folder
f = pathlib.Path(__file__)
shutil.copy(__file__, str(folder / f.name))

# build 'split', defines sbj which are affected
if one_vs_many:
    grp_split_idx = n_sbj - 1
else:
    grp_split_idx = int(n_sbj / 2)

arba.data.Split.fix_order(ft.sbj_list)
split = arba.data.Split({'grp_null': ft.sbj_list[:grp_split_idx],
                         'grp_eff': ft.sbj_list[grp_split_idx:]})

# go
with ft.loaded(split_eff_grp_list=[(split, effect, 'grp_eff')]):
    # write features to block img
    for feat_idx, feat in enumerate(ft.feat_list):
        img = nib.Nifti1Image(ft.data[:, :, :, :, feat_idx], ft.ref.affine)
        img.to_filename(str(folder / f'{feat}.nii.gz'))

    # agglomerative clustering
    sg_hist = arba.seg_graph.SegGraphBayes(file_tree=ft, split=split,
                                           verbose=True)
    sg_hist.reduce_to(1, verbose=True)
    merge_record = sg_hist.merge_record

    # save
    f = folder / 'sg_hist.p.gz'
    file.save(sg_hist, f)

    # build effect mask
    effect.mask.to_nii(folder / 'effect_mask.nii.gz')

    # build maha img
    maha = sg_hist.get_max_lower_bnd_array()
    maha_img = nib.Nifti1Image(maha, ft.ref.affine)
    maha_img.to_filename(str(folder / 'maha.nii.gz'))

    # print auc
    auc = effect.get_auc(maha, ft.mask)
    with open(str(folder / 'auc.txt'), 'w') as f:
        print(f'auc: {auc:.4f}', file=f)
    print(f'auc: {auc:.4f}')

    # # save hierarchical segmentation
    # arg_list = list()
    # merge_record = sg_hist.merge_record
    # merge_record.to_nii(f_out=folder / 'seg_hier.nii.gz', n=100)

    # build background image
    x = ft.data[:, :, :, :, 0].mean(axis=3)
    img = nib.Nifti1Image(x, ft.ref.affine)
    f_bg = folder / f'{ft.feat_list[0]}_sbj_mean.nii.gz'
    img.to_filename(str(f_bg))

    # find node which is closest to the effect region (via dice score)
    node, d_max = merge_record.get_node_max_dice(effect.mask)
    print(f'node: {node} d_max: {d_max}')

    node_set = {node}
    node_set |= nx.ancestors(merge_record, node)
    while True:
        node_next = list(merge_record.neighbors(node))
        if not node_next:
            break
        node = max(node_next)
        node_set.add(node)

    # plot maha(0) vs size
    sg, _ = merge_record.resolve_hist(file_tree=ft, split=split)
    # arba.plot.size_v_cdf_mu_bayes(sg, mask=effect.mask, mask_label='% effect')
    # arba.plot.save_fig(f_out=folder / 'size_v_cdf.pdf')

    # examine node of interest
    f_out = folder / f'localize_effect.pdf'
    with PdfPages(f_out) as pdf:
        for node in sorted(node_set):
            reg = merge_record.resolve_node(node=node,
                                            file_tree=ft,
                                            split=split)

            mask = reg.pc_ijk.to_mask(shape=ft.ref.shape)
            ax = arba.plot.plot_delta(mask=mask, reg=reg,
                                      mask_target=effect.mask,
                                      f_bg=f_bg,
                                      feat_list=ft.feat_list,
                                      grp1='grp_eff',
                                      feat_xylim=((0, 1), (.0001, .001)),
                                      delta_xylim=((-.2, .2), (-.0004, .0004)))
            ax[1].ticklabel_format(style='sci', axis='y')

            fig = plt.gcf()
            sens, spec = effect.get_sens_spec(estimate=mask, mask=ft.mask)
            num_obs = mask.sum() * n_sbj
            fig.suptitle(
                f'num_obs: {num_obs}, sens: {sens:.2f}, spec: {spec:.2f}')
            fig.set_size_inches(10, 10)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()

    arba.plot.size_v_norm_95_mu_bayes(sg, mask=effect.mask,
                                      mask_label='% effect')
    arba.plot.save_fig(f_out=folder / 'size_v_lower_bnd.pdf')
