import os
import shlex
import subprocess
import tempfile

import nibabel as nib
import numpy as np
from .permute import PermuteBase


class PermuteTFCE(PermuteBase):
    def __init__(self, *args, h=2, e=.5, c=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.e = e
        self.c = c

    def run_split(self, split, **kwargs):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        t2 = self.get_t2(split)
        t2_tfce = apply_tfce(t2, h=self.h, e=self.e, c=self.c)
        return t2_tfce


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