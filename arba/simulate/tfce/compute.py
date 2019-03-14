import os
import shlex
import subprocess
import tempfile

import nibabel as nib
import numpy as np


def to_file(x, tag=''):
    f = tempfile.NamedTemporaryFile(suffix=f'{tag}.nii.gz').name
    img_mask = nib.Nifti1Image(x, affine=np.eye(4))
    img_mask.to_filename(f)
    return f


def apply_tfce(x, **kwargs):
    """ applies tfce to an array, deletes files """

    # get input / output files
    f_x = to_file(x)
    f_out = tempfile.NamedTemporaryFile(suffix='_tfce.nii.gz').name

    # compute
    apply_tfce_file(f_in=f_x, f_out=f_out, **kwargs)

    # cleanup
    x_tfce = nib.load(f_out).get_data()
    os.remove(f_out)
    os.remove(f_x)

    return x_tfce


def apply_tfce_file(f_in, f_out=None, h=2, e=.5, c=6):
    # get f_out
    if f_out is None:
        f_out = tempfile.NamedTemporaryFile(suffix='_tfce.nii.gz').name
    # call randomise
    cmd = f'fslmaths {f_in} -tfce {h} {e} {c} {f_out}'

    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    return f_out
