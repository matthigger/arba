import os
import pathlib
import tempfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba.region import FeatStat, FeatStatSingle
from arba.space import PointCloud
from mh_pytools.parallel import run_par_fnc


def get_temp_file(*args, **kwargs):
    h, f = tempfile.mkstemp(*args, **kwargs)
    os.close(h)
    return pathlib.Path(f)


def compute_t2_per_ijk(ijk, x, sbj_cmp_list):
    num_sbj = x.shape[0]
    t2 = np.empty(num_sbj)

    fs = FeatStat.from_array(x.T)
    for sbj_idx, (sbj_cmp_bool, sbj_x) in enumerate(zip(sbj_cmp_list, x)):
        # compare to all but self
        _fs = fs
        if sbj_cmp_bool:
            _fs -= FeatStatSingle(sbj_x)

        # compute t_sq
        delta = sbj_x - _fs.mu
        t2[sbj_idx] = np.sqrt(delta @ _fs.cov_inv @ delta)

    return ijk, t2


def prep_files(ft_dict, f_data=None, grp_cmp_list=None, par_flag=False,
               **kwargs):
    """ build nifti of input data

    tfce requires a data cube [i, j, k, sbj_idx].  this function writes such
    a nifti file.

    Args:
        ft_dict (dict): keys are grp labels, values are file_trees
        grp_cmp_list (list): list of grp labels in comparison set
        par_flag (bool): toggles parallel computation

    Returns:
        f_data (Path): output file of nifti cube
        sbj_idx_dict (dict): keys are sbj, values are idx in datacube
    """
    grp_list = sorted(ft_dict.keys())

    if grp_cmp_list is None:
        grp_cmp_list = grp_list

    grp0 = grp_list[0]
    for grp in grp_list[1:]:
        ft0 = ft_dict[grp0]
        ft = ft_dict[grp]
        assert ft0.ref == ft.ref, 'space mismatch'
        assert not set(ft0.sbj_list) & set(ft.sbj_list), 'case overlap'
        assert np.allclose(ft0.mask.shape, ft.mask.shape), 'shape mismatch'
        assert np.allclose(ft0.mask, ft.mask), 'mask mistmatch'

    # get mask
    mask = ft_dict[grp0].mask

    # build mapping of sbj to idx
    sbj_list = list()
    for grp in grp_list:
        sbj_list += ft_dict[grp].sbj_list
    sbj_idx_dict = {sbj: idx for idx, sbj in enumerate(sbj_list)}

    # build data cube
    for ft in ft_dict.values():
        ft.load(load_data=True, load_ijk_fs=False)
    data_list = [ft_dict[grp].data for grp in grp_list]
    x = np.concatenate(data_list, axis=3)
    for ft in ft_dict.values():
        ft.unload()

    # build sbj_cmp_idx
    sbj_cmp_set = set().union(*[ft_dict[grp].sbj_list for grp in grp_cmp_list])
    sbj_cmp_list = [sbj in sbj_cmp_set for sbj in sbj_list]

    # prepare arguments
    arg_list = list()
    for ijk in PointCloud.from_mask(mask):
        i, j, k = ijk
        _x = x[i, j, k, :, :]
        arg_list.append({'ijk': ijk, 'x': _x, 'sbj_cmp_list': sbj_cmp_list})

    # compute t-squared per voxel (across entire population)
    t2 = np.zeros(x.shape[:4])
    if par_flag:
        res = run_par_fnc(compute_t2_per_ijk, arg_list=arg_list)
        for ijk, _t2 in res:
            i, j, k = ijk
            t2[i, j, k, :] = _t2
    else:
        for d in tqdm(arg_list, desc='compute t2 per ijk'):
            ijk, _t2 = compute_t2_per_ijk(**d)
            i, j, k = ijk
            t2[i, j, k, :] = _t2

    # ensure all t2 are positive
    _mask = np.broadcast_to(mask.T, t2.T.shape).T
    assert np.all(t2[_mask] > 0), 'invalid t2, must be positive'

    # get filenames
    if f_data is None:
        f_data = get_temp_file(suffix='_t2_tfce.nii.gz')
        f_mask = str(f_data).replace('_tfce.nii.gz', '_mask.nii.gz')
    else:
        folder = pathlib.Path(f_data).parent
        folder.mkdir(exist_ok=True)
        f_mask = folder / 'tfce_mask.nii.gz'

    # save data
    img_data = nib.Nifti1Image(t2, affine=ft0.ref.affine)
    img_data.to_filename(str(f_data))

    # save mask
    img_mask = nib.Nifti1Image(mask.astype(np.uint8), affine=ft0.ref.affine)
    img_mask.to_filename(str(f_mask))

    return f_data, f_mask, sbj_idx_dict, grp_list
