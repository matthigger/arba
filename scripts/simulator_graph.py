import os
import shlex
import subprocess
from collections import namedtuple, defaultdict

import nibabel as nib
from statsmodels.stats.multitest import multipletests

from mh_pytools import file
from pnl_segment.seg_graph import SegGraph
from pnl_segment.space import Mask, get_ref

performance = namedtuple('performance', ('sens', 'spec', 'dice', 'auc'))


class RegMahaDm:
    def __init__(self, pc_ijk, maha, pval):
        self.pc_ijk = pc_ijk
        self.maha = maha
        self.pval = pval


def compute_arba(folder, alpha):
    sg_hist_train = file.load(folder / 'sg_hist_train.p.gz')
    sg_arba_train = sg_hist_train.cut_greedy_pval(alpha)

    # reload sg_arba with separate data (not used to build segmentation)
    ft_dict_pval = file.load(folder / 'ft_dict_pval.p.gz')

    sg_hist = sg_hist_train.from_file_tree_dict(ft_dict_pval)
    sg_arba = sg_hist.cut_from_cut(sg_arba_train)

    file.save(sg_arba_train, folder / 'sg_arba_train.p.gz')
    file.save(sg_arba, folder / 'sg_arba.p.gz')
    file.save(sg_hist, folder / 'sg_hist.p.gz')


def compute_tfce(folder):
    # run tfce of wmaha
    f_vba = folder / f_stat_template.format(method='vba')
    f_tfce = folder / f_stat_template.format(method='tfce')
    cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
    subprocess.Popen(shlex.split(cmd)).wait()

    # build seg_graph_tfce (needed to compute dice)
    f_vba_sg = folder / 'sg_vba.p.gz'
    sg_vba = file.load(f_vba_sg)

    maha_tfce = nib.load(str(f_tfce)).get_data()
    sg_tfce = SegGraph()
    for reg in sg_vba.nodes:
        # get pval from reg using modified mahalanobis
        ijk = tuple(next(iter(reg.pc_ijk)))
        maha = maha_tfce[ijk]
        pval = reg.get_pval(maha=maha)

        # build dummy region, add to tfce segmentation graph
        reg_dm = RegMahaDm(pc_ijk=reg.pc_ijk, maha=maha, pval=pval)
        sg_tfce.add_node(reg_dm)
    file.save(sg_tfce, folder / 'sg_tfce.p.gz')


def get_perf(folder, method, alpha, f_detect_stat, effect):
    # get ref space
    f_vba = folder / f_stat_template.format(method='vba')
    ref = get_ref(f_vba)

    # load active mask
    mask_active = Mask.from_nii(folder / 'active_mask.nii.gz')

    # load sg of given method
    f_sg = folder / f_sg_template.format(method=method)
    sg = file.load(f_sg)

    # determines if significant
    pval_list = [reg.pval for reg in sg.nodes]
    if not pval_list:
        pval_list = [1]
    is_sig_vec = multipletests(pval_list, alpha=alpha, method='holm')[0]
    sig_reg_set = {r for r, is_sig in zip(sg.nodes, is_sig_vec) if is_sig}

    def is_sig(reg):
        return reg in sig_reg_set

    x = sg.to_array(fnc=is_sig, shape=ref.shape)

    dice = effect.get_dice(x)
    sens, spec = effect.get_sens_spec(x, mask_active)

    # compute auc
    x = sg.to_array(fnc=lambda x: x.pval, shape=ref.shape)
    auc = effect.get_auc(x, mask=mask_active)

    # output stats to file
    with open(str(f_detect_stat), 'a') as f:
        s_list = [f'{method}:',
                  f'    dice: {dice:.3f}',
                  f'    sens: {sens:.3f}',
                  f'    spec: {spec:.3f}',
                  f'    num_reg: {len(sg)}']
        f.write('\n'.join(s_list) + '\n\n')

    f_out = folder / f'detected_{method}.nii.gz'
    img = nib.Nifti1Image(x, affine=ref.affine)
    img.to_filename(str(f_out))

    return performance(sens=sens, spec=spec, dice=dice, auc=auc)


def run(folder, method_set, compute_arba_flag=False, alpha=.05, **kwargs):
    # load effect
    f_eff = folder / 'effect.p.gz'
    effect = file.load(f_eff)

    if compute_arba_flag:
        compute_arba(folder, alpha=alpha)

    if 'tfce' in method_set:
        compute_tfce(folder)

    # delete old f_detect_stat
    f_detect_stat = folder / 'detect_stats.txt'
    if f_detect_stat.exists():
        os.remove(str(f_detect_stat))

    # compute performance of each method
    method_perf_dict = defaultdict(list)
    for method in method_set:
        perf = get_perf(folder, method,
                        alpha=alpha,
                        f_detect_stat=f_detect_stat,
                        effect=effect)
        method_perf_dict[method].append(perf)

    return effect.snr, method_perf_dict


if __name__ == '__main__':
    from pnl_data.set.cidar_post import folder
    from mh_pytools import parallel

    folder_out = folder / 'synth_data'

    alpha = .05
    method_set = {'vba', 'arba', 'tfce'}
    par_flag = True
    recompute_arba = True

    f_stat_template = 'wmaha_{method}.nii.gz'
    f_sg_template = 'sg_{method}.p.gz'

    f_out = folder_out / 'performance_stats.p.gz'

    # find relevant folders, build inputs to run()
    arg_list = list()
    for folder in folder_out.glob('*snr*'):
        if not folder.is_dir():
            continue
        arg_list.append({'folder': folder,
                         'method_set': method_set,
                         'compute_arba_flag': recompute_arba,
                         'alpha': alpha})

    # run per folder
    if par_flag:
        res = parallel.run_par_fnc(run, arg_list)
        snr_method_perf_tree = {snr: perf_dict for snr, perf_dict in res}

    else:
        snr_method_perf_tree = dict()
        for d in arg_list:
            snr, perf_dict = run(**d)
            snr_method_perf_tree[snr] = perf_dict
    # save
    file.save(snr_method_perf_tree, f_out)
