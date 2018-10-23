from collections import defaultdict

import nibabel as nib
from tqdm import tqdm

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from pnl_segment.simulate.mask import Mask
from pnl_segment.simulate.simulator import get_f_img_dict
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import shlex
from pnl_segment.adaptive.part_graph_factory import part_graph_factory

folder_out = folder / '2018_Oct_23_10_20AM45'
sim, sbj_effect, sbj_health, obj = file.load(folder_out / 'sim_split.p.gz')
folder_fill = folder_out / '_no_effect'
f_stat_template = '_{obj}_{method}.nii.gz'
method_list = ['vba', 'arba']
f_out = folder_out / 'snr_auc.p.gz'
f_fs = folder / 'fs' / '01193' / 'aparc.a2009s+aseg_in_dti.nii.gz'
mask_active = Mask.from_nii(f_nii=folder_fill / 'mask_active.nii.gz')


def get_auc(effect, f_stat, f_fill=None, save=True):
    # get mask of positions to be filled (all with 0 in f_stat)
    img_stat = nib.load(str(f_stat))
    stat = img_stat.get_data()

    # fill
    if f_fill is not None:
        mask_fill = Mask.from_img(img_stat).negate()
        fill = mask_fill.apply_from_nii(f_fill)
        stat = mask_fill.insert(stat, fill, add=False)

        if save:
            f_out = str(f_stat).replace('.nii.gz', '_fill.nii.gz')
            img = nib.load(str(f_stat))
            img = nib.Nifti1Image(stat, img.affine)
            img.to_filename(f_out)

    return effect.get_auc(stat, mask_active)

def fnc(reg):
    return -reg.obj

# get mask of img
snr_auc_dict = defaultdict(list)
folder_missing_list = list()
for folder in tqdm(folder_out.glob('*snr*'), desc='experiment'):
    if not folder.is_dir():
        continue

    effect = file.load(folder / 'effect.p.gz')
    for method in method_list:
        f = f_stat_template.format(obj=obj, method=method)
        f_stat = folder / f
        f_fill = folder_fill / f

        auc = get_auc(effect, f_stat, f_fill)

        snr_auc_dict[method].append((effect.snr, auc))

    # build tfce img + compute auc
    f_vba = folder / f_stat_template.format(obj=obj, method='vba_fill')
    f_tfce = folder / f_stat_template.format(obj=obj, method='tfce')
    cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
    subprocess.Popen(shlex.split(cmd)).wait()

    auc = get_auc(effect, f_tfce)
    snr_auc_dict['tfce'].append((effect.snr, auc))

    # build rba img + compute auc
    f_img_dict, _ = get_f_img_dict(folder)
    pg = part_graph_factory(obj='max_maha', f_img_dict=f_img_dict)
    pg.combine_by_reg(f_fs)
    f_rba = folder / f_stat_template.format(obj=obj, method='rba')
    pg.to_nii(f_rba, fnc=fnc, ref=f_vba)

    auc = get_auc(effect, f_stat=f_rba)
    snr_auc_dict['rba'].append((effect.snr, auc))

# save
file.save(snr_auc_dict, f_out)

# load
snr_auc_dict = file.load(f_out)
sns.set()
plt.subplots(1, 1)
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log")
for method, snr_auc_list in snr_auc_dict.items():
    snr = [x[0] for x in snr_auc_list]
    auc = [x[1] for x in snr_auc_list]
    plt.scatter(snr, auc, label=method, alpha=.3)
plt.legend()
# plt.gca().set_xscale('log')
plt.gca().set_xlim(left=min(snr), right=max(snr))
plt.xlabel('snr')
plt.ylabel('auc')
plt.show()