import nibabel as nib
import numpy as np

from pnl_data import path


def split_label(f_fs, folder_out, label_str='', get_label=None, idx_keep=None):
    """ splits a freesurfer nii into many binary maps (per region)

    Args:
        f_fs (path): input freesurfer segmentation
        folder_out (path): path to folder where output masks will be stored
        label_str (str): str to add to label
        get_label (fnc): fnc which accepts idx and returns file label
        idx_keep (fnc): fnc which returns bool when applied to idx, if False
                        the idx is skipped (see region.valid.is_valid)

    Returns:
        f_fs_dict (dict): keys are idx, values are files which were written
    """

    if get_label is None:
        def get_label(folder_out, idx):
            return folder_out / f'{label_str}_{idx:.0f}.nii.gz'

    # read in freesurfer image
    img_fs = nib.load(str(f_fs))
    x = img_fs.get_data()

    f_fs_dict = dict()
    for idx in np.unique(x.flatten()):
        if idx_keep is not None and not idx_keep(idx):
            # idx_keep fnc passed and this idx has idx_keep(idx) = False
            continue

        # build mask
        mask = (x == idx).astype(np.uint8)
        img_mask = nib.Nifti1Image(mask, img_fs.affine)

        # write and record mask
        f_out = get_label(folder_out, idx)
        f_out.parent.mkdir(exist_ok=True, parents=True)
        img_mask.to_filename(str(f_out))
        f_fs_dict[idx] = f_out

    return f_fs_dict
