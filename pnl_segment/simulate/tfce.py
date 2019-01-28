import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from pnl_segment.space import Mask, PointCloud


def prep_files(ft_tuple, f_data=None, ft_effect_dict=dict(), mask=None,
               harmonize=False, **kwargs):
    """ build nifti of input data

    tfce requires a data cube [i, j, k, sbj_idx].  this function writes such
    a nifti file.

    Args:
        ft_tuple (tuple): two FileTree objects
        f_data (str or Path): output file of data cube
        ft_effect_dict (dict): keys are file trees (which should be ft0 or f1)
                               values are effects to apply to data cube.
                               useful for simulation
        mask : constrains output datacube
        harmonize (bool): toggles whether data per group is harmonized (in
                          initial partitioning)

    Returns:
        f_data (Path): output file of nifti cube
        sbj_idx_dict (dict): keys are sbj, values are idx in datacube
    """
    ft0, ft1 = ft_tuple
    assert ft0.feat_list == ft1.feat_list, 'feature mismatch'
    assert ft0.ref == ft1.ref, 'space mismatch'
    assert not set(ft0.sbj_iter) & set(ft1.sbj_iter), 'case overlap'
    # todo: support multivariate with Mahalanobis projection
    assert len(ft0.feat_list) == 1, 'only scalar features currently supported'

    # get constants from file trees
    ref = ft0.ref

    # init file
    if f_data is None:
        h, f_data = tempfile.mkstemp(suffix='tfce.nii.gz')
        os.close(h)
        f_data = pathlib.Path(f_data)

    # build mapping of sbj to idx
    sbj_list0 = sorted(ft0.sbj_feat_file_tree.keys())
    sbj_list1 = sorted(ft1.sbj_feat_file_tree.keys())
    sbj_idx_dict = {sbj: idx for idx, sbj in enumerate(sbj_list0)}
    sbj_idx_dict.update({sbj: idx + len(sbj_list0)
                         for idx, sbj in enumerate(sbj_list1)})
    grp0_bool = np.array([True] * len(sbj_list0) + [False] * len(sbj_list1))
    grp1_bool = np.logical_not(grp0_bool)
    grp_bool = grp0_bool, grp1_bool

    # get mask and f_mask
    f_mask = str(f_data).replace('.nii.gz', 'mask.nii.gz')
    if mask is None:
        # build mask as intersection of masks in each file tree
        mask = np.logical_and(ft0.mask, ft1.mask)
        mask.to_nii(f_out=f_mask)
    elif isinstance(mask, str) or isinstance(mask, pathlib.Path):
        # load it
        f_mask = pathlib.Path(mask)
        mask = Mask.from_nii(f_mask)
    elif isinstance(mask, Mask) or isinstance(mask, PointCloud):
        assert mask.ref == ref, 'mask reference mismatch'
        if isinstance(mask, PointCloud):
            # convert PointCloud to Mask
            mask = mask.to_mask()
        mask.to_nii(f_out=f_mask)
    else:
        raise TypeError('mask must be PointCloud, Mask or path to a nii')

    # build data array to temp file
    shape = (*ft0.ref.shape, len(sbj_idx_dict), len(ft0.feat_list))
    # todo: this init should be replaced by a broadcast...
    _mask = np.zeros(shape).astype(bool)
    not_mask = np.logical_not(mask)
    for i, j in np.ndindex(*shape[-2:]):
        _mask[:, :, :, i, j] = not_mask
    x = np.ma.MaskedArray(np.empty(shape), mask=_mask, fill_value=0)
    for ft in (ft0, ft1):
        for sbj in ft.sbj_iter:
            for feat_idx, feat in enumerate(ft0.feat_list):
                # file
                sbj_idx = sbj_idx_dict[sbj]
                f = ft.sbj_feat_file_tree[sbj][feat]

                # store
                x[:, :, :, sbj_idx, feat_idx] = nib.load(str(f)).get_data()

    # harmonize
    if harmonize:
        # compute averages per group (and total average)
        mu0 = x[:, :, :, grp_bool[0], :].mean(axis=(0, 1, 2, 3))
        mu1 = x[:, :, :, grp_bool[1], :].mean(axis=(0, 1, 2, 3))
        n0, n1 = len(sbj_list0), len(sbj_list1)
        mu = (mu0 * n0 + mu1 * n1) / (n0 + n1)

        # compute and apply offsets
        del0 = mu - mu0
        del1 = mu - mu1

        x[:, :, :, grp_bool[0], :] += del0
        x[:, :, :, grp_bool[1], :] += del1

    # add effect
    for ft, effect in ft_effect_dict.items():
        # get view of data corresponding to given file tree
        grp_idx = ft_tuple.index(ft)
        y = x[:, :, :, grp_bool[grp_idx], :]

        # apply
        y[effect.mask, :, :] += effect.mean

    # build and write data
    img_data = nib.Nifti1Image(x, affine=ref.affine)
    img_data.to_filename(str(f_data))

    return f_data, f_mask, sbj_idx_dict


def run_tfce(f_data, nm, folder, num_perm=500, f_mask=None, **kwargs):
    """ wrapper around randomise, interfaces at file level

    Args:
        f_data (str or Path): data cube (see build_data())
        nm (tuple): lengths of each group
        folder (str or Path): output of tfce
        num_perm (int): number of permutations
        f_mask (str or Path): mask

    Returns:
        f_tfce_pval (Path): file to nii of pval
    """

    f_design_ttest2 = shutil.which('design_ttest2')
    f_randomise = shutil.which('randomise')

    assert f_design_ttest2, 'design_ttest2 not found, FSL install / path?'
    assert f_randomise, 'randomise not found, FSL install / path?'

    # write design.mat, design.con
    f_design = folder / 'design'
    n, m = nm
    cmd = f'{f_design_ttest2} {f_design} {n} {m}'
    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    # call randomise
    cmd = f'randomise -i {f_data} -o {folder}/one_minus ' + \
          f'-d {f_design}.mat -t {f_design}.con ' + \
          f'-n {num_perm} -T --quiet'
    if f_mask is not None:
        cmd += f' -m {f_mask}'
    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    # rewrite output pval' (fsl does 1-p ...)
    f_pval_list = list()
    for idx in [1, 2]:
        f_pval_flip = folder / f'one_minus_tfce_corrp_tstat{idx}.nii.gz'
        f_pval = folder / f'tfce_corrp_tstat{idx}_p.nii.gz'
        img_p_flip = nib.load(str(f_pval_flip))
        img_p = nib.Nifti1Image(1 - img_p_flip.get_data(), img_p_flip.affine)
        img_p.to_filename(str(f_pval))
        f_pval_list.append(f_pval)

    return f_pval_list


def compute_tfce(ft_tuple, alpha=.05, folder=None, **kwargs):
    """ computes tfce by calling randomise

    Args:
        ft_tuple (tuple): two FileTree objects
        alpha (float): family wise error rate
        f_sig (str or Path): output file of significant voxels (bool)
        folder (str or Path): folder to put files

    Returns:
        f_sig (Path): output file of significant voxels
    """
    if folder is None:
        folder = pathlib.Path(tempfile.mkdtemp())

    # prep data cube
    f_data, f_mask, _ = prep_files(ft_tuple, **kwargs)

    # run tfce
    nm = tuple(len(ft) for ft in ft_tuple)
    f_pval_list = run_tfce(f_data, folder=folder, f_mask=f_mask, nm=nm,
                           **kwargs)

    # load pval, apply FWER significance threshold and save binary image
    f_sig_list = list()
    for idx, f_pval in enumerate(f_pval_list):
        # find mask of significant voxels
        img_pval = nib.load(str(f_pval))
        sig = img_pval.get_data() <= alpha
        img_sig = nib.Nifti1Image(sig.astype(np.uint8), img_pval.affine)

        # save
        f_sig = folder / f'mask_sig_tfce{idx}.nii.gz'
        img_sig.to_filename(str(f_sig))
        f_sig_list.append(f_sig)

    return folder, f_sig_list
