import shlex
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from pnl_segment.plot.snr_vs_dice import snr_vs_dice
from pnl_segment.space.mask import Mask, get_ref

spec = .05
obj = 'maha'
folder_out = folder / '2018_Nov_12_08_16AM31'
f_stat_template = '{obj}_{method}.nii.gz'
method_list = ['vba', 'arba', 'rba', 'truth']
tfce = False
f_out_pdf = folder_out / 'snr_auc.pdf'


# mask_whole = Mask.from_nii(f_nii=folder_fill / 'active_mask.nii.gz')


def fill(f_stat, f_fill, mask, f_out):
    stat = nib.load(str(f_stat)).get_data()
    fill = nib.load(str(f_fill)).get_data()
    for ijk in mask.to_point_cloud():
        fill[ijk] = stat[ijk]
    img = nib.Nifti1Image(fill, nib.load(str(f_stat)).affine)
    img.to_filename(str(f_out))


# get mask of img
sg_eff_dict = defaultdict(list)
method_snr_auc_dict = defaultdict(list)
folder_missing_list = list()
for folder in tqdm(folder_out.glob('*snr*'), desc='experiment'):
    if not folder.is_dir():
        continue

    # load active mask (of experiment)
    mask = Mask.from_nii(folder / 'active_mask.nii.gz')

    # build tfce img + compute auc
    f_vba = folder / f_stat_template.format(obj=obj, method='vba')
    if tfce:
        f_tfce = folder / f_stat_template.format(obj=obj, method='tfce')
        cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
        subprocess.Popen(shlex.split(cmd)).wait()
        method_list.append('tfce')

    # compute auc
    eff = file.load(folder / 'effect.p.gz')
    for method in method_list:
        f_sg = folder / f'sg_{method}.p.gz'
        if f_sg.exists():
            sg = file.load(f_sg)
            sg_eff_dict[method].append((sg, eff))

            def is_sig(reg):
                return reg.pval < (spec / len(sg))

            sg.to_nii(f_out=folder / f'detected_{method}.nii.gz',
                      ref=get_ref(f_vba),
                      fnc=is_sig)
        f = folder / f_stat_template.format(obj=obj, method=method)
        try:
            x = nib.load(str(f)).get_data()
        except FileNotFoundError:
            continue
        auc = eff.get_auc(x, mask=mask)

        method_snr_auc_dict[method, eff.snr].append(auc)

# plot snr vs dice
f_out = folder_out / 'snr_dice.pdf'
with PdfPages(str(f_out)) as pdf:
    snr_vs_dice(sg_eff_dict)
    plt.gca().set_xscale('log')
    pdf.savefig(plt.gcf())
    plt.close()

# save
f_out = folder_out / 'snr_auc.p.gz'
file.save(method_snr_auc_dict, f_out)

# plot
method_set = set(method for method, _ in method_snr_auc_dict.keys())
sns.set(font_scale=1.2)
plt.subplots(1, 1)
plt.gca().set_xscale("log", nonposx='clip')
for method in method_set:
    snr_auc_dict = [(snr, np.mean(auc_list))
                    for (_method, snr), auc_list in method_snr_auc_dict.items()
                    if _method == method]
    snr_auc = sorted(snr_auc_dict)
    snr = [x[0] for x in snr_auc]
    auc = [x[1] for x in snr_auc]
    plt.plot(snr, auc, label=method)
plt.legend()
# plt.gca().set_xscale('log')
plt.gca().set_xlim(left=min(snr), right=max(snr))
plt.xlabel('maha / voxel')
plt.ylabel('auc')

fig = plt.gcf()
fig.set_size_inches(10, 7)
with PdfPages(str(f_out_pdf)) as pdf:
    pdf.savefig(fig)
