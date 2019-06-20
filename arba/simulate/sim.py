import pathlib
import tempfile

import networkx as nx
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import arba
from arba.permute import apply_tfce_file, apply_ptfce
from mh_pytools import file


def print_score_arba(sg_hist, f_out):
    maha = sg_hist.get_max_lower_bnd_array()
    maha_img = nib.Nifti1Image(maha, sg_hist.file_tree.ref.affine)
    maha_img.to_filename(str(f_out))


def get_maha(reg):
    fs0, fs1 = reg.fs_dict.values()
    delta = fs1.mu - fs0.mu
    cov = (fs1 + fs0).cov
    return np.sqrt(delta @ np.linalg.inv(cov) @ delta)


def print_score_vba_rba(sg_hist, f_out_vba, split, f_rba=None, f_out_rba=None):
    assert sg_hist.file_tree.is_loaded, 'file_tree must be loaded'

    file_tree = sg_hist.file_tree

    sg_vba, _, _ = next(sg_hist.merge_record.get_iter_sg(file_tree=file_tree,
                                                         split=split))

    sg_vba.to_nii(f_out=f_out_vba, fnc=get_maha)

    if f_rba is not None:
        sg_vba.merge_by_atlas(f_region=f_rba, skip_zero=False)
        sg_rba = sg_vba
        sg_rba.to_nii(f_out=f_out_rba, fnc=get_maha)


def print_score_tfce(f_in, f_tfce_out=None, f_ptfce_out=None, f_mask=None):
    if f_tfce_out:
        apply_tfce_file(f_in, f_out=f_tfce_out)
    if f_ptfce_out:
        apply_ptfce(f_in, f_out=f_ptfce_out, f_mask=f_mask)


def compute_auc(folder, effect, mask, f_out):
    s_glob = 'score*.nii.gz'

    auc_dict = dict()
    for f in folder.glob(s_glob):
        label = f.name.split('.')[0].split('_')[-1]
        score = nib.load(str(f)).get_data()

        auc_dict[label] = effect.get_auc(score, mask)

    label_auc_list = sorted(auc_dict.items(), key=lambda x: x[1], reverse=True)
    for label, auc in label_auc_list:
        print(f'label: {label} auc: {auc:.4f}')

    if f_out is not None:
        with open(str(f_out), 'w') as f:
            for label, auc in label_auc_list:
                print(f'label: {label} auc: {auc:.4f}', file=f)


def run(file_tree, split, split_eff_grp_list=None, folder=None, verbose=True,
        f_rba=None,
        print_per_sbj=False, print_hier_seg=False, print_lower_bnd=False,
        print_eff_zoom=False, save_sg_hist=False, compute_auc_flag=False):
    # build folder
    if folder is None:
        folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        print(folder)
        folder.mkdir(exist_ok=True, parents=True)
    else:
        folder = pathlib.Path(folder)

    with file_tree.loaded(split_eff_grp_list=split_eff_grp_list):
        # agglomerative clustering
        sg_hist = arba.seg_graph.SegGraphBayes(file_tree=file_tree,
                                               split=split,
                                               verbose=verbose)
        sg_hist.reduce_to(1, verbose=verbose)
        merge_record = sg_hist.merge_record

        # save
        if save_sg_hist:
            f = folder / 'sg_hist.p.gz'
            file.save(sg_hist, f)

        # print effect mask
        if split_eff_grp_list is not None:
            _, effect, grp_effect = split_eff_grp_list[0]
            f_effect_mask = folder / 'effect_mask.nii.gz'
            effect.mask.to_nii(f_effect_mask)

        if f_rba is None:
            f_rba = f_effect_mask

        print_score_arba(sg_hist, f_out=folder / 'score_arba.nii.gz')
        print_score_vba_rba(sg_hist, f_rba=f_rba,
                            f_out_vba=folder / 'score_vba.nii.gz',
                            f_out_rba=folder / 'score_rba.nii.gz',
                            split=split)
        print_score_tfce(folder / 'score_vba.nii.gz',
                         f_tfce_out=folder / 'score_tfce.nii.gz',
                         f_ptfce_out=None,
                         f_mask=str(file_tree.mask.to_nii()))

        if compute_auc_flag:
            assert split_eff_grp_list is not None
            compute_auc(folder, effect=effect, mask=file_tree.mask,
                        f_out=folder / 'auc.txt')

        # save hierarchical segmentation
        if print_hier_seg:
            merge_record.to_nii(f_out=folder / 'seg_hier.nii.gz', n=100)

        if print_per_sbj:
            for feat_idx, feat in enumerate(file_tree.feat_list):
                img = nib.Nifti1Image(file_tree.data[:, :, :, :, feat_idx],
                                      file_tree.ref.affine)
                img.to_filename(str(folder / f'{feat}.nii.gz'))

        if print_lower_bnd:
            assert split_eff_grp_list is not None
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
    else:
        f_out = pathlib.Path(f_out)

    file_tree = sg_hist.file_tree
    assert file_tree.is_loaded
    split, effect, grp_effect = file_tree.split_eff_grp_list[0]

    file_tree = sg_hist.file_tree
    merge_record = sg_hist.merge_record

    # build background image
    x = file_tree.data[:, :, :, :, 0].mean(axis=3)
    img = nib.Nifti1Image(x, file_tree.ref.affine)
    f_bg = f_out.parent / f'{file_tree.feat_list[0]}_all_sbj_mean.nii.gz'
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
            num_vox = mask.sum()
            fig.suptitle(
                f'num_vox: {num_vox}, sens: {sens:.2f}, spec: {spec:.2f}')
            fig.set_size_inches(10, 10)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
