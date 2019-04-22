import shlex
import subprocess

import numpy as np


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


if __name__ == '__main__':
    f_z = '/tmp/tmpn_mumzwx.nii.gz'
    f_mask = '/tmp/tmp1vqzz15l.nii.gz'

    print(compute_smooth(f_z, f_mask))