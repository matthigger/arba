import pathlib
import tempfile

import nibabel as nib
import numpy as np

import arba
from mh_pytools import file

# file tree
n_sbj = 15
mu = (0, 0)
cov = np.eye(2)
shape = 5, 5, 5

node_to_examine = 227

# effect
mask = np.zeros(shape)
mask[2:4, 2:4, 2:4] = 1
offset = (3, -1)

# bayes threshold
alpha = .05

np.random.seed(1)

# build file tree
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)
_folder = folder / 'data_orig'
_folder.mkdir(exist_ok=True, parents=True)
ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                             folder=_folder)

# build effect
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

    # run bayes on each volume, record mv_norm and lower_bnd
    arg_list = list()
    merge_record = sg_hist.merge_record
    merge_record.to_nii(f_out=folder / 'seg_hier.nii.gz', n=100)

    # build background image
    x = ft.data[:, :, :, :, 0].mean(axis=3)
    img = nib.Nifti1Image(x, ft.ref.affine)
    f_bg = folder / f'{ft.feat_list[0]}_sbj_mean.nii.gz'
    img.to_filename(str(f_bg))

    # examine node of interest
    reg = merge_record.resolve_node(node=node_to_examine,
                                    file_tree=ft,
                                    split=split)
    grp_mu_cov_dict = {grp: (fs.mu, fs.cov) for grp, fs in reg.fs_dict.items()}
    f_out = folder / f'node_{node_to_examine}.pdf'
    mask = reg.pc_ijk.to_mask(shape=ft.ref.shape)
    arba.plot.plot_delta(mask=mask, grp_mu_cov_dict=grp_mu_cov_dict, f_bg=f_bg,
                         feat_list=ft.feat_list, f_out=f_out,
                         feat_xylim=((-3, 13), (-3, 13)),
                         delta_xylim=((-5, 5), (-5, 5)))
