import shlex
import subprocess
from collections import defaultdict

import nibabel as nib
from scipy.stats import chi2

from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder
from pnl_segment.region.stats import BenHochYek
from pnl_segment.seg_graph import SegGraph
from pnl_segment.space import Mask, get_ref

spec = .05
obj = 'maha'
folder_out = folder / 'rad_5'
f_stat_template = '{obj}_{method}.nii.gz'
f_sg_template = 'sg_{method}.p.gz'
method_set = {'vba', 'arba', 'rba', 'truth', 'tfce'}
par_flag = True


class regMahaDm:
    def __init__(self, pc_ijk, maha, pval):
        self.pc_ijk = pc_ijk
        self.maha = maha
        self.pval = pval


def get_dice_auc(folder):
    method_snr_auc_dict = defaultdict(list)
    method_snr_dice_dict = defaultdict(list)
    method_snr_sens_spec_dict = defaultdict(list)

    # load active mask (of experiment)
    mask_active = Mask.from_nii(folder / 'active_mask.nii.gz')
    eff = file.load(folder / 'effect.p.gz')

    # build tfce img + compute auc
    f_vba = folder / f_stat_template.format(obj=obj, method='vba')
    if 'tfce' in method_set:
        f_tfce = folder / f_stat_template.format(obj=obj, method='tfce')
        cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
        subprocess.Popen(shlex.split(cmd)).wait()

        # build seg_graph_tfce (needed to compute dice)
        f_vba_sg = folder / 'sg_vba.p.gz'
        sg_vba = file.load(f_vba_sg)

        maha_tfce = nib.load(str(f_tfce)).get_data()
        sg_tfce = SegGraph()
        for reg in sg_vba.nodes:
            ijk = next(iter(reg.pc_ijk))
            maha = maha_tfce[tuple(ijk)]
            pval = chi2.sf(maha, df=2)
            reg_dm = regMahaDm(pc_ijk=reg.pc_ijk, maha=maha, pval=pval)
            sg_tfce.add_node(reg_dm)
        file.save(sg_tfce, folder / 'sg_tfce.p.gz')

    # compute dice / auc
    for method in method_set:
        # compute dice
        f_sg = folder / f_sg_template.format(method=method)
        sg = file.load(f_sg)

        # determines if significant
        pval_list = [reg.pval for reg in sg.nodes]
        thresh = BenHochYek(pval_list, sig=spec)

        def is_sig(reg):
            return reg.pval < thresh

        x = sg.to_array(fnc=is_sig)

        dice = eff.get_dice(x)
        ss = eff.get_sens_spec(x, mask_active)
        method_snr_sens_spec_dict[method, eff.snr].append(ss)

        f_out = folder / f'detected_{method}.nii.gz'
        img = nib.Nifti1Image(x, affine=get_ref(f_vba).affine)
        img.to_filename(str(f_out))

        method_snr_dice_dict[method, eff.snr].append(dice)

        # compute auc
        f = folder / f_stat_template.format(obj=obj, method=method)
        try:
            x = nib.load(str(f)).get_data()
        except FileNotFoundError:
            continue
        auc = eff.get_auc(x, mask=mask_active)

        method_snr_auc_dict[method, eff.snr].append(auc)

    return method_snr_dice_dict, method_snr_auc_dict, method_snr_sens_spec_dict, eff


# get mask of img
arg_list = list()
for folder in folder_out.glob('*snr*'):
    if not folder.is_dir():
        continue
    arg_list.append({'folder': folder})

method_snr_auc_dict = defaultdict(list)
method_snr_dice_dict = defaultdict(list)
method_snr_sens_spec_dict = defaultdict(list)
eff_dict = defaultdict(list)
if par_flag:
    res = parallel.run_par_fnc(get_dice_auc, arg_list)
    for d_dice, d_auc, d_ss, eff in res:
        for k in d_dice.keys():
            method_snr_auc_dict[k].append(d_auc[k])
            method_snr_dice_dict[k].append(d_dice[k])
            method_snr_sens_spec_dict[k].append(d_ss[k])
        eff_dict[eff.snr].append(eff)

else:
    for d in arg_list:
        d_dice, d_auc, d_ss, eff = get_dice_auc(**d)
        for k in d_dice.keys():
            method_snr_auc_dict[k].append(d_auc[k])
            method_snr_dice_dict[k].append(d_dice[k])
            method_snr_sens_spec_dict[k].append(d_ss[k])
        eff_dict[eff.snr].append(eff)

# save
f_out = folder_out / 'snr_auc_dice.p.gz'
file.save((
          method_snr_auc_dict, method_snr_dice_dict, method_snr_sens_spec_dict,
          eff_dict), f_out)
