import pathlib
import tempfile

import nibabel as nib
import numpy as np

from arba.simulate.tfce.prep_files import prep_files
from arba.simulate.tfce.run import run_tfce


def compute_tfce(ft_dict, alpha=.05, folder=None, **kwargs):
    """ computes tfce by calling randomise

    Args:
        ft_dict (dict): keys are grp labels, values are file_trees
        alpha (float): family wise error rate
        folder (str or Path): folder to put files

    Returns:
        f_sig (Path): output file of significant voxels
    """
    if folder is None:
        folder = pathlib.Path(tempfile.mkdtemp())

    # prep data cube
    f_data, f_mask, _, grp_list = prep_files(ft_dict, **kwargs)

    # run tfce
    nm = tuple(len(ft_dict[grp]) for grp in grp_list)
    kwargs['f_data'] = f_data
    f_pval_list = run_tfce(folder=folder, f_mask=f_mask, nm=nm,
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