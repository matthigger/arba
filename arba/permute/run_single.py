import pathlib
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from scipy.ndimage import label

from arba.plot import size_v_t2, size_v_pval, save_fig
from mh_pytools import file


def get_folder(fnc):
    def wrapped(folder=None, **kwargs):
        if folder is None:
            folder = pathlib.Path(tempfile.TemporaryDirectory().name)

        # print output folder to cmd line
        print(f'{fnc.__name__} output folder: {folder}')

        return fnc(folder=folder, **kwargs)

    return wrapped


def run_print_single(**kwargs):
    sg_hist, folder = run_single(**kwargs)

    plot_tree(sg_hist=sg_hist, folder=folder)

    seg = print_seg(sg_hist=sg_hist, folder=folder)

    # print_stats(seg=seg, sg_hist=sg_hist)


@get_folder
def plot_tree(sg_hist=None, folder=None):
    f_sg_hist = folder / 'sg_hist.p.gz'
    if f_sg_hist.exists() and sg_hist is None:
        sg_hist = file.load(f_sg_hist)
    else:
        assert sg_hist is not None, 'either sg_hist xor folder'

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
         sg_hist (SegGraphHistT2):
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
    if f_sg_hist.exists() and sg_hist is None:
        assert sg_hist is None, 'either sg_hist xor folder'
        sg_hist = file.load(f_sg_hist)
    else:
        assert sg_hist is not None, 'either sg_hist xor folder'

    t2 = sg_hist.get_max_t2_array()
    t2_idx, num_reg = label(t2)

    t2_idx_list = list()
    for _t2_idx in range(num_reg):
        _t2 = t2[t2_idx == _t2_idx][0]
        t2_idx_list.append((_t2, t2_idx))
    t2_idx_list = sorted(t2_idx_list)

    seg = np.zeros(t2.shape)
    for sort_idx, (_t2, _t2_idx) in enumerate(t2_idx_list):
        seg[_t2_idx == t2_idx] = sort_idx

    # print seg
    seg_img = nib.Nifti1Image(seg, sg_hist.file_tree.ref.affine)
    seg_img.to_filename(str(folder / 'seg.nii.gz'))

    # print t2
    pval_img = nib.Nifti1Image(t2, sg_hist.file_tree.ref.affine)
    pval_img.to_filename(str(folder / 't2.nii.gz'))

    return seg


@get_folder
def print_stats(seg=None, sg_hist=None, folder=None, num_reg=10):
    sns.set(font_scale=1.2)

    try:
        _sg_hist = file.load(folder / 'sg_hist.p.gz')
        _seg = nib.load(str(folder / 'seg.nii.gz')).get_data()
        sg_hist = _sg_hist
        seg = _seg
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
