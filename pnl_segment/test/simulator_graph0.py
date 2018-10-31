import shlex
import subprocess
from collections import defaultdict

import nibabel as nib
from tqdm import tqdm

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from pnl_segment.simulate.mask import Mask

obj = 'max_maha'
folder_out = folder / '2018_all_1'
folder_fill = folder_out / 'healthy_run000'
f_stat_template = '{obj}_{method}.nii.gz'
method_list = ['vba', 'arba', 'rba', 'truth']
tfce = True
# mask_whole = Mask.from_nii(f_nii=folder_fill / 'active_mask.nii.gz')


def fill(f_stat, f_fill, mask, f_out):
    stat = nib.load(str(f_stat)).get_data()
    fill = nib.load(str(f_fill)).get_data()
    for ijk in mask:
        fill[ijk] = stat[ijk]
    img = nib.Nifti1Image(fill, nib.load(str(f_stat)).affine)
    img.to_filename(str(f_out))


# get mask of img
method_snr_auc_dict = defaultdict(list)
folder_missing_list = list()
for folder in tqdm(folder_out.glob('*snr*'), desc='experiment'):
    if not folder.is_dir():
        continue

    # load active mask (of experiment)
    mask = Mask.from_nii(folder / 'active_mask.nii.gz')

    # # build filled stat
    # for method in method_list:
    #     if method == 'tfce':
    #         continue
    #     f = f_stat_template.format(obj=obj, method=method)
    #     f_stat = folder / f
    #     f_fill = folder_fill / f
    #     f_out = str(f_stat).replace('.nii.gz', '_fill.nii.gz')
    #
    #     fill(f_stat, f_fill, mask, f_out)

    # build tfce img + compute auc
    if tfce:
        f_vba = folder / f_stat_template.format(obj=obj, method='vba')
        f_tfce = folder / f_stat_template.format(obj=obj, method='tfce')
        cmd = f'fslmaths {f_vba} -tfce 2 0.5 6 {f_tfce}'
        subprocess.Popen(shlex.split(cmd)).wait()
        method_list.append('tfce')

    # compute auc
    eff = file.load(folder / 'effect.p.gz')
    for method in method_list:
        f = folder / f_stat_template.format(obj=obj, method=method)
        try:
            x = nib.load(str(f)).get_data()
        except FileNotFoundError:
            continue
        auc = eff.get_auc(x, mask=mask)

        method_snr_auc_dict[method, eff.snr].append(auc)

# save
f_out = folder_out / 'snr_auc.p.gz'
file.save(method_snr_auc_dict, f_out)
