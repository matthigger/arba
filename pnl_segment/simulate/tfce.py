import pathlib
import shlex
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from pnl_segment.space import Mask, PointCloud


def run_tfce(ft0, ft1, folder=None, num_perm=500, mask=None):
    """ runs tfce given two FileTrees

    note: file trees must contain only a single feature

    Args:
        ft0 (FileTree): file tree of first group
        ft1 (FileTree): file tree of second group
        folder (str or Path): output of tfce
        num_perm (int): number of permutations
        mask: mask of active voxels (str, Path to nii or Mask / PointCloud). if
              none, defaults to intersection of all voxels observed in all img

    Returns:
        f_tfce_pval (Path): file to nii of pval
    """

    f_design_ttest2 = shutil.which('design_ttest2')
    f_randomise = shutil.which('randomise')

    assert f_design_ttest2, 'design_ttest2 not found, FSL install / path?'
    assert f_randomise, 'randomise not found, FSL install / path?'

    assert ft0.ref == ft1.ref, 'space mismatch'
    assert ft0.feat_list == ft1.feat_list, 'feature mismatch'
    assert len(ft0.feat_list) == 1, 'tfce requires scalar input'

    # get constants from file trees
    feat = ft0.feat_list[0]
    ref = ft0.ref

    # init folder
    if folder is None:
        folder = pathlib.Path(tempfile.mkdtemp(prefix='run_tfce_'))

    # get f_mask
    if mask is None:
        # build mask as intersection of masks in each file tree
        mask = np.logical_and(ft0.mask, ft1.mask)

    if isinstance(mask, str) or isinstance(mask, pathlib.Path):
        f_mask = pathlib.Path(mask)
    elif isinstance(mask, Mask) or isinstance(mask, PointCloud):
        assert mask.ref == ref, 'mask reference mismatch'
        if isinstance(mask, PointCloud):
            # convert PointCloud to Mask
            mask = mask.to_mask()

        # write mask to output file
        f_mask = folder / 'mask.nii.gz'
        mask.to_nii(f_out=f_mask)
    else:
        raise TypeError('mask must be PointCloud, Mask or path to a nii')

    # build mapping of sbj to idx, each sbj is assumed unique
    sbj_list0 = sorted(ft0.sbj_feat_file_tree.keys())
    sbj_list1 = sorted(ft1.sbj_feat_file_tree.keys())
    sbj_idx_dict = {sbj: idx for idx, sbj in enumerate(sbj_list0)}
    sbj_idx_dict.update({sbj: idx + len(sbj_list0)
                         for idx, sbj in enumerate(sbj_list1)})

    # build data array to temp file
    shape = (*ft0.ref.shape, len(sbj_idx_dict))
    x = np.empty(shape)
    for ft in (ft0, ft1):
        for sbj in ft.sbj_iter:
            # get img
            idx = sbj_idx_dict[sbj]
            f = ft.sbj_feat_file_tree[sbj][feat]
            img = nib.load(f)

            # store
            x[:, :, :, idx] = img.get_data()

    # build and write data
    f_data = folder / 'data.nii.gz'
    img_data = nib.Nifti1Image(x, affine=ref.affine)
    img_data.to_filename(str(f_data))

    # write sbj_idx_dict (not needed for tfce, but good for debug post mortem)
    f_sbj_to_idx = folder / 'sbj_idx_dict.txt'
    idx_sbj_list = sorted((idx, sbj) for sbj, idx in sbj_idx_dict.items())
    with open(str(f_sbj_to_idx), 'w') as f:
        for idx, sbj in idx_sbj_list:
            print(f'{idx} {sbj}', file=f)

    # write design.mat, design.con
    f_design = folder / 'design'
    cmd = f'{f_design_ttest2} {f_design} {len(ft0)} {len(ft1)}'
    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    # call randomise
    cmd = f'randomise -i {f_data} -o {folder}/one_minus ' + \
          f'-d {f_design}.mat -t {f_design}.con ' + \
          f'-m {f_mask} -n {num_perm} -T'
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
