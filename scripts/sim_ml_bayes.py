import pathlib
import tempfile

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import arba
from mh_pytools import file

# file tree
n_sbj = 10
mu = (0, 0)
cov = np.eye(2) * 2
shape = 10, 10, 10

# effect
mask = np.zeros(shape)
mask[0:5, 0:5, 0:5] = 1
offset = (1, 1)

np.random.seed(1)

# build file tree
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)
_folder = folder / 'data_orig'
_folder.mkdir(exist_ok=True, parents=True)
ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                             folder=_folder)

# build effect
mask = arba.space.Mask(mask, ref=ft.ref)
effect = arba.simulate.Effect(mask=mask, offset=offset)

# build 'split', defines sbj which are affected
arba.data.Split.fix_order(ft.sbj_list)
half = int(n_sbj / 2)
split = arba.data.Split({False: ft.sbj_list[:half],
                         True: ft.sbj_list[half:]})

# go
with ft.loaded(split_eff_list=[(split, effect)]):
    # write features to block img
    for feat_idx, feat in enumerate(ft.feat_list):
        img = nib.Nifti1Image(ft.data[:, :, :, :, feat_idx], ft.ref.affine)
        img.to_filename(str(folder / f'{feat}.nii.gz'))

    # agglomerative clustering
    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=ft, split=split)
    sg_hist.reduce_to(1)

    # save
    f = folder / 'sg_hist.p.gz'
    file.save(sg_hist, f)

    # save hierarchical segmentation
    arg_list = list()
    merge_record = sg_hist.merge_record
    merge_record.to_nii(f_out=folder / 'seg_hier.nii.gz', n=100)

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

    # examine node of interest
    f_out = folder / f'localize_effect.pdf'
    with PdfPages(f_out) as pdf:
        for node in sorted(node_set):
            reg = merge_record.resolve_node(node=node,
                                            file_tree=ft,
                                            split=split)
            grp_mu_cov_dict = reg.bayes_mu()

            mask = reg.pc_ijk.to_mask(shape=ft.ref.shape)
            arba.plot.plot_delta(mask=mask, grp_mu_cov_dict=grp_mu_cov_dict,
                                 mask_target=effect.mask,
                                 f_bg=f_bg,
                                 feat_list=ft.feat_list,
                                 feat_xylim=((-3, 3), (-3, 3)),
                                 delta_xylim=((-3, 3), (-3, 3)))

            fig = plt.gcf()
            sens, spec = effect.get_sens_spec(estimate=mask, mask=ft.mask)
            num_obs = mask.sum() * n_sbj
            fig.suptitle(
                f'num_obs: {num_obs}, sens: {sens:.2f}, spec: {spec:.2f}')
            fig.set_size_inches(10, 10)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
