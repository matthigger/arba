import copy
import shutil

import nibabel as nib
import numpy as np

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder
from pnl_segment.simulate import Simulator, compute_tfce


def run_vba_rba_tfce(folder, alpha, write_outfile=True):
    folder_image = folder / 'image'
    folder_tfce = folder / 'tfce'

    s_mask_sig = 'mask_sig_{method}.nii.gz'
    ft_dict = file.load(folder / 'ft_dict.p.gz')
    sg_hist_seg = file.load(folder / 'sg_hist_seg.p.gz')
    effect = file.load(folder / 'effect_grp_effect.p.gz')

    # build file tree of entire dataset (no folds needed)
    ft0, ft1 = ft_dict[Simulator.grp_null], ft_dict[Simulator.grp_effect]

    # compute tfce
    f_sig = folder_image / s_mask_sig.format(method='tfce')
    _, f_sig_list = compute_tfce((ft0, ft1), alpha=alpha, folder=folder_tfce,
                                 f_data=folder_tfce / 'maha.nii.gz')
    shutil.copy(f_sig_list[1], str(f_sig))

    # compute vba + rba
    ft0.load(load_ijk_fs=True, load_data=False)
    ft1.load(load_ijk_fs=True, load_data=False)
    sg_vba_test, _, _ = next(iter(sg_hist_seg))
    sg_vba = sg_vba_test.from_file_tree_dict(ft_dict)
    sg_dict = {'vba': sg_vba, 'rba': copy.deepcopy(sg_vba)}
    sg_dict['rba'].combine_by_reg(f_rba)

    # output masks of detected volumes
    for method, sg in sg_dict.items():
        sg_sig = sg.get_sig(alpha=alpha)
        sg_sig.to_nii(f_out=folder_image / s_mask_sig.format(method=method),
                      ref=ft0.ref,
                      fnc=lambda r: np.uint8(1),
                      background=np.uint8(0))

    # compute performance
    method_ss_dict = dict()
    for method in ['arba', 'vba', 'tfce', 'rba']:
        f_estimate = folder_image / s_mask_sig.format(method=method)
        estimate = nib.load(str(f_estimate)).get_data()
        method_ss_dict[method] = effect.get_sens_spec(estimate=estimate,
                                                      mask=ft0.mask)
    if write_outfile:
        f_out = folder / 'detect_stats.txt'
        with open(str(f_out), 'w') as f:
            for method, (sens, spec) in sorted(method_ss_dict.items()):
                print(f'{method}: sens {sens:.2f} spec {spec:.2f}', file=f)

    return round(effect.maha, 4), method_ss_dict


if __name__ == '__main__':
    from collections import defaultdict
    from tqdm import tqdm

    par_flag = True
    alpha = .05
    f_rba = folder / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
    folder = folder / 'test_cv_tmp'
    f_out = folder / 'performance_stats.p.gz'

    # find relevant folders, build inputs to run()
    arg_list = list()
    for folder in sorted(folder.glob('*maha*')):
        if not folder.is_dir():
            continue
        arg_list.append({'folder': folder,
                         'alpha': alpha})

    # run per folder
    fnc = lambda: defaultdict(list)
    maha_method_perf_tree = defaultdict(fnc)
    desc = 'compute rba, vba, tfce per experiment'
    if par_flag:
        res = parallel.run_par_fnc(run_vba_rba_tfce, arg_list, desc=desc)
        for maha, method_ss_dict in res:
            for method, ss in method_ss_dict.items():
                maha_method_perf_tree[maha][method].append(ss)

    else:
        for d in tqdm(arg_list, desc=desc):
            maha, method_ss_dict = run_vba_rba_tfce(**d)
            for method, ss in method_ss_dict.items():
                maha_method_perf_tree[maha][method].append(ss)
    # save
    file.save(dict(maha_method_perf_tree), f_out)
