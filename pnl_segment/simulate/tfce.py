import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from pnl_segment.seg_graph import FileTree


def get_temp_file(*args, **kwargs):
    h, f = tempfile.mkstemp(*args, **kwargs)
    os.close(h)
    return pathlib.Path(f)


def prep_files(ft_tuple, ft_effect_dict=dict(), mask=None, harmonize=False):
    """ build nifti of input data

    tfce requires a data cube [i, j, k, sbj_idx].  this function writes such
    a nifti file.

    Args:
        ft_tuple (tuple): two FileTree objects
        ft_effect_dict (dict): keys are file trees (which should be ft0 or f1)
                               values are effects to apply to data cube.
                               useful for simulation
        mask (Mask): constrains output datacube
        harmonize (bool): toggles whether data per group is harmonized (in
                          initial partitioning)

    Returns:
        f_data (Path): output file of nifti cube
        sbj_idx_dict (dict): keys are sbj, values are idx in datacube
    """
    ft0, ft1 = ft_tuple
    assert ft0.ref == ft1.ref, 'space mismatch'
    assert not set(ft0.sbj_list) & set(ft1.sbj_list), 'case overlap'

    # get mask
    if mask is None:
        assert np.allclose(ft0.mask, ft1.mask), 'mask mistmatch'
        mask = ft0.mask

    # ensure data is loaded
    for ft in ft_tuple:
        ft.load(mask=mask)

    # harmonize
    if harmonize:
        FileTree.harmonize_via_add(ft_tuple, apply=True)

    # apply effects
    for ft, effect in ft_effect_dict.items():
        effect.apply_to_file_tree(ft)

    # build mapping of sbj to idx
    sbj_list = ft0.sbj_list + ft1.sbj_list
    sbj_idx_dict = {sbj: idx for idx, sbj in enumerate(sbj_list)}

    # build data cube
    x = np.concatenate((ft0.data, ft1.data), axis=3)

    # apply mask (mask is
    assert np.allclose(mask.shape, x.shape[:3]), 'mask shape mismatch'
    _mask = np.broadcast_to(mask.T, x.T.shape).T
    x[np.logical_not(_mask)] = 0

    # mahalanobis or scalar reduce
    # todo: currently just tosses a dimension ... should apply maha
    x = np.squeeze(x, axis=4)

    # save data
    f_data = get_temp_file(suffix='_tfce.nii.gz')
    img_data = nib.Nifti1Image(x, affine=ft0.ref.affine)
    img_data.to_filename(str(f_data))

    # save mask
    f_mask = str(f_data).replace('_tfce.nii.gz', '_mask.nii.gz')
    img_mask = nib.Nifti1Image(mask.astype(np.uint8), affine=ft0.ref.affine)
    img_mask.to_filename(str(f_mask))

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

    # build folder
    folder = pathlib.Path(folder)
    folder.mkdir(exist_ok=True)

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
