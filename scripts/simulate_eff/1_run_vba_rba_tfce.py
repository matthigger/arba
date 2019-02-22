import copy
import shutil

import nibabel as nib
import numpy as np

from arba.simulate import Simulator, compute_tfce
from mh_pytools import file, parallel


def get(folder, f):
    """ there may be multiple copies of f within the folder (different
    subfolders).  any of them are valid, we just grab the first one
    """
    f = next(folder.glob(f'**/{f}'))
    return file.load(str(f))


def run_vba_rba_tfce(folder, alpha, write_outfile=True, f_rba=None):
    s_mask_sig = 'mask_sig_{method}.nii.gz'

    ft_dict = get(folder, 'ft_dict.p.gz')
    sg_hist_seg = get(folder, 'sg_hist_seg.p.gz')
    effect = get(folder, 'effect_grp_effect.p.gz')

    # build file tree of entire dataset (no folds needed)
    ft0, ft1 = ft_dict[Simulator.grp_null], ft_dict[Simulator.grp_effect]

    # compute tfce
    folder_tfce = folder / 'tfce'
    folder_tfce.mkdir(exist_ok=True)
    f_sig = folder_tfce / s_mask_sig.format(method='tfce')
    _, f_sig_list = compute_tfce((ft0, ft1), alpha=alpha, folder=folder_tfce,
                                 f_data=folder_tfce / 'maha.nii.gz')
    shutil.copy(f_sig_list[1], str(f_sig))

    # compute vba + rba
    ft0.load(load_ijk_fs=True, load_data=False)
    ft1.load(load_ijk_fs=True, load_data=False)
    sg_vba_test, _, _ = next(iter(sg_hist_seg))
    sg_vba = sg_vba_test.from_file_tree_dict(ft_dict)
    sg_dict = {'vba': sg_vba}
    if f_rba is not None:
        sg_dict['rba'] = copy.deepcopy(sg_vba)
        sg_dict['rba'].combine_by_reg(f_rba)

    # output masks of detected volumes
    for method, sg in sg_dict.items():
        folder_method = folder / method
        folder_method.mkdir(exist_ok=True)
        sg_sig = sg.get_sig(alpha=alpha)
        sg_sig.to_nii(f_out=folder_method / s_mask_sig.format(method=method),
                      ref=ft0.ref,
                      fnc=lambda r: np.uint8(1),
                      background=np.uint8(0))

    # compute performance
    method_ss_dict = dict()
    for folder_method in folder.glob('*'):
        if not folder_method.is_dir():
            continue

        method = folder_method.name
        f_estimate = folder_method / s_mask_sig.format(method=method)
        estimate = nib.load(str(f_estimate)).get_data()
        method_ss_dict[method] = effect.get_sens_spec(estimate=estimate,
                                                      mask=ft0.mask)

    if write_outfile:
        f_out = folder / 'detect_stats.txt'
        with open(str(f_out), 'w') as f:
            for method, (sens, spec) in sorted(method_ss_dict.items()):
                print(f'{method}: sens {sens:.2f} spec {spec:.2f}', file=f)

    return (round(effect.maha, 4), int(effect.mask.sum())), method_ss_dict


if __name__ == '__main__':
    from collections import defaultdict
    from tqdm import tqdm
    from pnl_data.set.hcp_100 import folder

    par_flag = True
    alpha = .05
    # f_rba = folder / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
    f_rba = None
    folder = folder / '2019_Feb_11_09_43_11'
    f_out = folder / 'performance_stats.p.gz'

    # find relevant folders, build inputs to run()
    arg_list = list()
    for folder in sorted(folder.glob('*maha*')):
        if not folder.is_dir():
            continue
        arg_list.append({'folder': folder,
                         'alpha': alpha,
                         'f_rba': f_rba})

    # run per folder
    fnc = lambda: defaultdict(list)
    mahasize_method_ss_tree = defaultdict(fnc)
    desc = 'compute rba, vba, tfce per experiment'
    if par_flag:
        res = parallel.run_par_fnc(run_vba_rba_tfce, arg_list, desc=desc)
        for maha_size, method_ss_dict in res:
            for method, ss in method_ss_dict.items():
                mahasize_method_ss_tree[maha_size][method].append(ss)

    else:
        for d in tqdm(arg_list, desc=desc):
            maha_size, method_ss_dict = run_vba_rba_tfce(**d)
            for method, ss in method_ss_dict.items():
                mahasize_method_ss_tree[maha_size][method].append(ss)
    # save
    file.save(dict(mahasize_method_ss_tree), f_out)
