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


def get_f_out(fnc, suffix='.nii.gz'):
    def wrapped(*args, f_out=None, **kwargs):
        if f_out is None:
            f_out = tempfile.NamedTemporaryFile(suffix=suffix)
        f_out = pathlib.Path(f_out)
        return fnc(*args, f_out=f_out, **kwargs)

    return wrapped


@get_f_out
def print_score_arba(sg_hist, f_out):
    maha = sg_hist.get_max_lower_bnd_array()
    maha_img = nib.Nifti1Image(maha, sg_hist.file_tree.ref.affine)
    maha_img.to_filename(str(f_out))
    return f_out


def compute_auc():
    raise NotImplementedError
    # print auc
    auc = effect.get_auc(maha, file_tree.mask)
    with open(str(folder / 'auc.txt'), 'w') as f:
        print(f'auc: {auc:.4f}', file=f)
    print(f'auc: {auc:.4f}')


def run(file_tree, split_eff_grp, folder=None, verbose=True,
        print_per_sbj=False, print_hier_seg=False, print_lower_bnd=False,
        print_eff_zoom=False):
    # build folder
    if folder is None:
        folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        print(folder)
        folder.mkdir(exist_ok=True, parents=True)
    else:
        folder = pathlib.Path(folder)

    split, effect, grp_effect = split_eff_grp

    with file_tree.loaded(split_eff_grp_list=[split_eff_grp]):
        # agglomerative clustering
        sg_hist = arba.seg_graph.SegGraphBayes(file_tree=file_tree,
                                               split=split,
                                               verbose=verbose)
        sg_hist.reduce_to(1, verbose=verbose)
        merge_record = sg_hist.merge_record

        # save
        f = folder / 'sg_hist.p.gz'
        file.save(sg_hist, f)

        # print effect mask
        effect.mask.to_nii(folder / 'effect_mask.nii.gz')

        print_score_arba(sg_hist, f_out=folder / 'score_arba.nii.gz')

        # save hierarchical segmentation
        if print_hier_seg:
            merge_record.to_nii(f_out=folder / 'seg_hier.nii.gz', n=100)

        if print_per_sbj:
            for feat_idx, feat in enumerate(file_tree.feat_list):
                img = nib.Nifti1Image(file_tree.data[:, :, :, :, feat_idx],
                                      file_tree.ref.affine)
                img.to_filename(str(folder / f'{feat}.nii.gz'))

        if print_lower_bnd:
            sg, _ = merge_record.resolve_hist(file_tree=file_tree, split=split)
            arba.plot.size_v_norm_95_mu_bayes(sg, mask=effect.mask,
                                              mask_label='% effect')
            arba.plot.save_fig(f_out=folder / 'size_v_lower_bnd.pdf')

        if print_eff_zoom:
            f_out = folder / f'effect_zoom.pdf'
            effect_zoom(sg_hist, f_out=f_out)


def effect_zoom(sg_hist, f_out):
    if f_out is None:
        f_out = tempfile.NamedTemporaryFile(suffix='effect_zoom.nii.gz')

    file_tree = sg_hist.file_tree
    merge_record = sg_hist.merge_record

    # build background image
    x = file_tree.data[:, :, :, :, 0].mean(axis=3)
    img = nib.Nifti1Image(x, file_tree.ref.affine)
    f_bg = folder / f'{file_tree.feat_list[0]}_all_sbj_mean.nii.gz'
    img.to_filename(str(f_bg))

    # find node which is closest to the effect region (via dice score)
    node, d_max = merge_record.get_node_max_dice(effect.mask)
    print(f'node: {node} d_max: {d_max}')

    # find sequence of nodes from voxel to whole region which include node
    node_set = {node}
    node_set |= nx.ancestors(merge_record, node)
    while True:
        node_next = list(merge_record.neighbors(node))
        if not node_next:
            break
        node = max(node_next)
        node_set.add(node)

    # for each node, 'plot_delta'
    with PdfPages(f_out) as pdf:
        for node in sorted(node_set):
            reg = merge_record.resolve_node(node=node,
                                            file_tree=file_tree,
                                            split=split)

            mask = reg.pc_ijk.to_mask(shape=file_tree.ref.shape)
            ax = arba.plot.plot_delta(mask=mask, reg=reg,
                                      mask_target=effect.mask,
                                      f_bg=f_bg,
                                      feat_list=file_tree.feat_list,
                                      grp1='grp_eff')
            ax[1].ticklabel_format(style='sci', axis='y')

            fig = plt.gcf()
            sens, spec = effect.get_sens_spec(estimate=mask,
                                              mask=file_tree.mask)
            num_obs = mask.sum() * n_sbj
            fig.suptitle(
                f'num_obs: {num_obs}, sens: {sens:.2f}, spec: {spec:.2f}')
            fig.set_size_inches(10, 10)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    # file tree
    n_sbj = 10
    mu = (0, 0)
    cov = np.eye(2) * 1
    shape = 10, 10, 10

    offset = (1, 1)

    fake_data = True
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
            effect = arba.simulate.Effect.from_fs_t2(fs=fs, t2=1,
                                                     mask=eff_mask,
                                                     u=offset)

        # mask file_tree (lowers compute needed)
        mask = binary_dilation(eff_mask, iterations=4)
        ft.mask = np.logical_and(mask, ft.mask)

    # build 'split', defines sbj which are affected
    if one_vs_many:
        grp_split_idx = n_sbj - 1
    else:
        grp_split_idx = int(n_sbj / 2)

    arba.data.Split.fix_order(ft.sbj_list)
    split = arba.data.Split({'grp_null': ft.sbj_list[:grp_split_idx],
                             'grp_eff': ft.sbj_list[grp_split_idx:]})

    # copy script to output folder
    f = pathlib.Path(__file__)
    shutil.copy(__file__, str(folder / f.name))

    split_eff_grp = (split, effect, 'grp_eff')

    run(file_tree=ft, split_eff_grp=split_eff_grp, folder=folder, verbose=True,
        print_per_sbj=True, print_hier_seg=True, print_lower_bnd=True,
        print_eff_zoom=True)
