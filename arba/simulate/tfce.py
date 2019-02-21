import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from arba.space import PointCloud


def get_temp_file(*args, **kwargs):
    h, f = tempfile.mkstemp(*args, **kwargs)
    os.close(h)
    return pathlib.Path(f)


def prep_files(ft_tuple, f_data=None):
    """ build nifti of input data

    tfce requires a data cube [i, j, k, sbj_idx].  this function writes such
    a nifti file.

    Args:
        ft_tuple (tuple): two FileTree objects, the first defines the compare
                          group (defines mu and cov in computation of
                          Mahalanobis distance)

    Returns:
        f_data (Path): output file of nifti cube
        sbj_idx_dict (dict): keys are sbj, values are idx in datacube
    """
    ft0, ft1 = ft_tuple
    assert ft0.ref == ft1.ref, 'space mismatch'
    assert not set(ft0.sbj_list) & set(ft1.sbj_list), 'case overlap'

    # get mask
    assert np.allclose(ft0.mask, ft1.mask), 'mask mistmatch'
    mask = ft0.mask

    # build mapping of sbj to idx
    sbj_list = ft0.sbj_list + ft1.sbj_list
    sbj_idx_dict = {sbj: idx for idx, sbj in enumerate(sbj_list)}

    # build data cube
    ft0.load(load_data=True, load_ijk_fs=False)
    ft1.load(load_data=True, load_ijk_fs=False)
    x = np.concatenate((ft0.data, ft1.data), axis=3)
    ft0.unload()
    ft1.unload()

    # apply mask
    assert np.allclose(mask.shape, x.shape[:3]), 'mask shape mismatch'

    # compute mahalanobis per voxel (across entire population)
    maha = np.zeros(x.shape[:4])
    for i, j, k in PointCloud.from_mask(mask):

        _x = x[i, j, k, :, :]
        for sbj_idx, sbj_x in enumerate(_x):
            # compare to all but self
            sbj_cmp_idx = np.ones(len(sbj_list)).astype(bool)
            sbj_cmp_idx[sbj_idx] = 0

            # get stats
            _x_cmp = _x[sbj_cmp_idx, :]
            mu = np.mean(_x_cmp, axis=0)
            cov = np.atleast_2d(np.cov(_x_cmp.T))
            cov_inv = np.linalg.pinv(cov)

            # compute t_sq
            delta = sbj_x - mu
            maha[i, j, k, sbj_idx] = np.sqrt(delta @ cov_inv @ delta)

    # ensure all maha are positive
    _mask = np.broadcast_to(mask.T, maha.T.shape).T
    assert np.all(maha[_mask] > 0), 'invalid maha, must be positive'

    # get filenames
    if f_data is None:
        f_data = get_temp_file(suffix='_maha_tfce.nii.gz')
        f_mask = str(f_data).replace('_tfce.nii.gz', '_mask.nii.gz')
    else:
        folder = pathlib.Path(f_data).parent
        folder.mkdir(exist_ok=True)
        f_mask = folder / 'tfce_mask.nii.gz'

    # save data
    img_data = nib.Nifti1Image(maha, affine=ft0.ref.affine)
    img_data.to_filename(str(f_data))

    # save mask
    img_mask = nib.Nifti1Image(mask.astype(np.uint8), affine=ft0.ref.affine)
    img_mask.to_filename(str(f_mask))

    return f_data, f_mask, sbj_idx_dict


def run_tfce(f_data, nm, folder, num_perm=100, f_mask=None):
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
    f_pval_list = run_tfce(f_data, folder=folder, f_mask=f_mask, nm=nm)

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
