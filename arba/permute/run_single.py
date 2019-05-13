import pathlib
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from arba.plot import size_v_t2, size_v_pval, save_fig
from mh_pytools import file
from scipy.ndimage import label


def get_folder(fnc):
    def wrapped(*args, folder=None, **kwargs):
        if folder is None:
            folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        print(f'{fnc.__name__} output folder: {folder}')
        fnc(*args, folder=folder, **kwargs)

    return wrapped


@get_folder
def run_print_single(**kwargs):
    sg_hist, folder = run_single(**kwargs)

    plot_tree(sg_hist, folder)

    seg = print_seg(sg_hist=sg_hist)

    # print_stats(seg=seg, sg_hist=sg_hist)


@get_folder
def plot_tree(sg_hist, folder=None):
    merge_record = sg_hist.merge_record
    file_tree = sg_hist.file_tree

    with file_tree.loaded():
        tree_hist, _ = merge_record.resolve_hist(file_tree=sg_hist.file_tree,
                                                 split=sg_hist.split)

    for plt_fnc in (size_v_t2, size_v_pval):
        plt.figure()
        plt_fnc(tree_hist)
        save_fig(folder / f'{plt_fnc.__name__}.pdf')


@get_folder
def run_single(sg_hist, folder=None, **kwargs):
    """ runs ARBA on a single sg_hist, prints segmentation and stats

    Args:
         sg_hist (SegGraphHistPval):
         folder (pathlib.Path):

    Returns:
        sg_hist (SegGraphHistory):
    """
    sg_hist.reduce_to(1, **kwargs)

    file.save(sg_hist, folder / 'sg_hist.p.gz')

    return sg_hist, folder


@get_folder
def print_seg(sg_hist=None, folder=None):
    f_sg_hist = folder / 'sg_hist.p.gz'
    if f_sg_hist.exists():
        assert sg_hist is None, 'either sg_hist xor folder'
        sg_hist = file.load(f_sg_hist)
    else:
        assert sg_hist is not None, 'either sg_hist xor folder'

    pval = sg_hist.get_min_pval_array()
    pval_idx, num_reg = label(pval)

    pval_idx_list = list()
    for _pval_idx in range(num_reg):
        _pval = pval[pval_idx == _pval_idx][0]
        pval_idx_list.append((_pval, pval_idx))
    pval_idx_list = sorted(pval_idx_list)

    seg = np.zeros(pval.shape)
    for sort_idx, (_pval, _pval_idx) in enumerate(pval_idx_list):
        seg[_pval_idx == pval_idx] = sort_idx

    # print seg
    seg_img = nib.Nifti1Image(seg, sg_hist.file_tree.ref.affine)
    seg_img.to_filename(str(folder / 'seg.nii.gz'))

    # print pval
    pval_img = nib.Nifti1Image(pval, sg_hist.file_tree.ref.affine)
    pval_img.to_filename(str(folder / 'pval.nii.gz'))

    return seg


@get_folder
def print_stats(seg=None, sg_hist=None, folder=None, num_reg=10):
    sns.set(font_scale=1.2)

    try:
        sg_hist = file.load(folder / 'sg_hist.p.gz')
        seg = nib.load(str(folder / 'seg.nii.gz')).get_data()
    except FileNotFoundError:
        assert (seg is not None) and (sg_hist is not None), \
            'seg and sg_hist required if folder not passed'

    file_tree = sg_hist.file_tree

    with file_tree.loaded():
        for reg_idx in range(num_reg):
            mask = seg == (reg_idx + 1)

        raise NotImplementedError
        x = file_tree.data[mask, :, :]
        sns.kdeplot(x)
