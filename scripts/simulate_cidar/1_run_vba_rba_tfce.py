import copy
import shlex
import subprocess

import nibabel as nib
import numpy as np

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder
from pnl_segment.seg_graph import SegGraph
from pnl_segment.space import Mask


class RegMahaDm:
    def __init__(self, pc_ijk, maha, pval):
        self.pc_ijk = pc_ijk
        self.maha = maha
        self.pval = pval


def compute_sg_tfce(sg_vba, mask, folder=None):
    # init temp files
    if folder is None:
        f_tfce = file.get_temp(suffix='.nii.gz')
        f_vba = file.get_temp(suffix='.nii.gz')
    else:
        f_tfce = str(folder / 'wmaha_tfce.nii.gz')
        f_vba = str(folder / 'wmaha_vba.nii.gz')

    # write vba maha to temp file, apply tfce
    sg_vba.to_nii(f_out=f_vba,
                  ref=mask.ref,
                  fnc=lambda r: r.maha,
                  background=0)
    cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
    subprocess.Popen(shlex.split(cmd)).wait()

    # read in tfce(maha), make dummy regions
    tfce_maha = nib.load(f_tfce).get_data()
    sg = SegGraph()
    for reg_vba in sg_vba.nodes:
        pc_ijk = reg_vba.pc_ijk
        maha = tfce_maha[next(iter(pc_ijk))]
        r = RegMahaDm(pc_ijk=pc_ijk,
                      maha=maha,
                      pval=reg_vba.get_pval(maha=maha))
        sg.add_node(r)

    return sg


def run_vba_rba_tfce(folder, alpha, write_outfile=True):
    s_mask_sig = 'mask_sig_{method}.nii.gz'

    ft_dict = file.load(folder / 'ft_dict.p.gz')
    sg_hist_test = file.load(folder / 'sg_hist.p.gz')
    mask = Mask.from_nii(folder / 'mask.nii.gz')
    effect = file.load(folder / 'effect.p.gz')

    for ft in ft_dict.values():
        ft.load(mask=mask)

    effect.apply_to_file_tree(ft_dict['grp_effect'])

    # initial element in __iter__ has one voxel per region (vba)
    sg_vba_test = next(iter(sg_hist_test))

    # build sg_vba and sg_rba
    sg_vba = sg_vba_test.from_file_tree_dict(ft_dict)
    sg_dict = {'vba': sg_vba, 'rba': copy.deepcopy(sg_vba)}
    sg_dict['rba'].combine_by_reg(f_rba)

    sg_dict['tfce'] = compute_sg_tfce(sg_vba, mask, folder)

    # output masks of detected volumes
    for method, sg in sg_dict.items():
        sg_sig = sg.is_sig(alpha=alpha)
        sg_sig.to_nii(f_out=folder / s_mask_sig.format(method=method),
                      ref=mask.ref,
                      fnc=lambda r: np.uint8(1),
                      background=np.uint8(0))

    # compute performance
    method_ss_dict = dict()
    for method in ['arba', 'vba', 'tfce', 'rba']:
        f_estimate = folder / s_mask_sig.format(method=method)
        estimate = nib.load(str(f_estimate)).get_data()
        method_ss_dict[method] = effect.get_sens_spec(estimate=estimate,
                                                      mask=mask)
    if write_outfile:
        f_out = folder / 'detect_stats.txt'
        with open(str(f_out), 'w') as f:
            for method, (sens, spec) in sorted(method_ss_dict.items()):
                print(f'{method}: sens {sens:.2f} spec {spec:.2f}', file=f)

    return effect.snr, method_ss_dict


if __name__ == '__main__':
    from collections import defaultdict

    par_flag = True
    alpha = .05
    f_rba = folder / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
    folder = folder / '2018_Dec_27_07_40AM00'
    f_out = folder / 'performance_stats.p.gz'

    # find relevant folders, build inputs to run()
    arg_list = list()
    for folder in folder.glob('*snr*'):
        if not folder.is_dir():
            continue
        arg_list.append({'folder': folder,
                         'alpha': alpha})

    # run per folder
    fnc = lambda: defaultdict(list)
    snr_method_perf_tree = defaultdict(fnc)
    if par_flag:
        res = parallel.run_par_fnc(run_vba_rba_tfce, arg_list)
        for snr, method_ss_dict in res:
            for method, ss in method_ss_dict.items():
                snr_method_perf_tree[snr][method].append(ss)

    else:
        for d in arg_list:
            snr, method_ss_dict = run_vba_rba_tfce(**d)
            for method, ss in method_ss_dict.items():
                snr_method_perf_tree[snr][method].append(ss)
    # save
    file.save(dict(snr_method_perf_tree), f_out)
