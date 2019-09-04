import pathlib
import shlex
import subprocess
import tempfile

import nibabel as nib
import numpy as np

f_run_ptfce = pathlib.Path(__file__).parent / 'run_ptfce.R'
Z_MAX = 20


def apply_ptfce(f_in, f_mask, f_out=None, sqrt=False):
    # get f_out
    if f_out is None:
        f_out = tempfile.NamedTemporaryFile(suffix='_ptfce.nii.gz').name

    img_in = nib.load(str(f_in))
    z = img_in.get_data()

    if sqrt:
        # useful for input img which are maha or t2
        z = np.sqrt(z)

    # https://github.com/spisakt/pTFCE/issues/3
    z[z > Z_MAX] = Z_MAX

    # write to file
    img_z = nib.Nifti1Image(z, img_in.affine)
    f_z = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
    img_z.to_filename(f_z)

    # compute smoothness via FSL
    smooth_dict = compute_smooth(f_z=f_z, f_mask=f_mask)
    v = int(smooth_dict['VOLUME'])
    r = smooth_dict['RESELS']

    # apply ptfce
    cmd = f'Rscript {f_run_ptfce} -i {f_z} -m {f_mask} -o {f_out} -v {v} -r {r}'
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL)

    proc.wait()

    return f_out


def compute_smooth(f_z, f_mask):
    cmd = f'smoothest -z {f_z} -m {f_mask}'
    s = subprocess.check_output(shlex.split(cmd))

    dict_out = dict()
    for line in str(s).split('\\n'):
        line_split = line.split(' ')
        if len(line_split) <= 1:
            continue
        elif len(line_split) == 2:
            val = float(line_split[1])
        else:
            val = np.array([float(x) for x in line_split[1:]])
        feat = line_split[0]
        dict_out[feat] = val

    return dict_out
