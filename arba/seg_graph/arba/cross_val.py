import pathlib

from arba.seg_graph.arba.prep import prep_arba
from arba.seg_graph.seg_graph_hist import SegGraphHistory
from mh_pytools import file


def run_arba_cv(ft_dict, folder=None, verbose=False, alpha=.05, **kwargs):
    """ runs arba (cross validation), optionally saves outputs.

    Args:
        ft_dict (dict): keys are population labels, values are FileTree
        folder (str or Path): output folder for experiment, if None will
                                   not save
        verbose (bool): toggles command line output
        alpha (float): false positive rate

    Returns:
        sg_arba_test (SegGraph): candidate regions (test data)
    """
    # split into segmentation + test data
    ft_dict_seg = dict()
    ft_dict_test = dict()
    for grp, ft in ft_dict.items():
        ft_dict_seg[grp], ft_dict_test[grp] = ft.split(p=.5)

    # prep each
    ft_dict_seg = prep_arba(ft_dict_seg, label='seg', folder=folder,
                            verbose=verbose, **kwargs)
    ft_dict_test = prep_arba(ft_dict_test, label='test', folder=folder,
                             verbose=verbose, **kwargs)
    prep_arba(ft_dict, label='', folder=folder, verbose=verbose,
              **kwargs)

    # build sg_hist_seg
    sg_hist_seg = SegGraphHistory(ft_dict=ft_dict_seg)

    # reduce
    sg_hist_seg.reduce_to(1, verbose=verbose, **kwargs)

    # determine candidate regions
    sg_arba_seg = sg_hist_seg.cut_greedy_sig(alpha=alpha)

    # swap data source for test data
    sg_arba_test = sg_arba_seg.from_ft_dict(ft_dict_test)
    sg_hist_test = sg_hist_seg.from_ft_dict(ft_dict_test)

    # determine which regions are sig
    sg_arba_test_sig = sg_arba_test.get_sig(alpha=alpha)

    # save
    if folder is not None:
        folder = pathlib.Path(folder)
        folder_save = folder / 'save'
        folder.mkdir(exist_ok=True, parents=True)

        file.save(sg_arba_seg, folder_save / 'sg_arba_seg.p.gz')
        file.save(sg_arba_test, folder_save / 'sg_arba_test.p.gz')

        file.save(sg_hist_seg, folder_save / 'sg_hist_seg.p.gz')
        file.save(sg_hist_test, folder_save / 'sg_hist_test.p.gz')
        file.save(sg_arba_test_sig, folder_save / 'sg_arba_test_sig.p.gz')

        # save sig mask
        ref = next(iter(ft_dict_seg.values())).ref
        sg_arba_test_sig.to_nii(folder / 'mask_sig_arba_cv.nii.gz',
                                ref=ref,
                                fnc=lambda r: 1,
                                background=0)

    return sg_arba_test
