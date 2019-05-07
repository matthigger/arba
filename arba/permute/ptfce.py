import multiprocessing
import os
import pathlib
import shlex
import subprocess
import tempfile
from time import sleep

import nibabel as nib
import numpy as np
import psutil
from tqdm import tqdm

from .permute import PermuteBase

f_run_ptfce = pathlib.Path(__file__).parent / 'run_ptfce.R'


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


class PermutePTFCE(PermuteBase):
    flag_max = True

    # memory needed to run each
    mem_buff = 5e9
    mem_per_run = 24e8
    Z_MAX = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # write mask to temp file
        self.f_mask = self.file_tree.mask.to_nii()

    def __del__(self):
        os.remove(str(self.f_mask))

    def run_split_max_multi(self, split_list, par_flag=False, verbose=False,
                            effect_split_dict=None, check_fs=1, **kwargs):
        """ kludge to ensure only 12 processors, R is memory hungry"""
        if not par_flag:
            return super().run_split_max_multi(split_list,
                                               par_flag=par_flag,
                                               verbose=verbose,
                                               effect_split_dict=effect_split_dict,
                                               **kwargs)

        mem_total = psutil.virtual_memory().available - self.mem_buff
        n_proc_max = np.floor(mem_total / self.mem_per_run)
        proc_split_file_dict = dict()
        pbar = tqdm(total=len(split_list),
                    disable=not verbose,
                    desc='max ptfce per split')
        while split_list or proc_split_file_dict:
            sleep(1 / check_fs)

            # collect data from finished jobs
            proc_to_rm = list()
            for proc, (split, file) in proc_split_file_dict.items():
                if proc.poll() is None:
                    continue

                # process finished, collect and store max stat
                proc_to_rm.append(proc)
                max_stat = nib.load(str(file)).get_data().max()
                self.split_stat_dict[split] = max_stat
                os.remove(str(file))

                # update pbar
                pbar.update(1)

            # remove finished processes
            for proc in proc_to_rm:
                del proc_split_file_dict[proc]

            # no more memory, dont start new job
            if len(proc_split_file_dict) >= n_proc_max:
                continue
            mem_available = psutil.virtual_memory().available - self.mem_buff
            if mem_available < self.mem_per_run:
                continue

            # no more cpu, dont start new job
            if len(proc_split_file_dict) > multiprocessing.cpu_count():
                continue

            # start new job
            if split_list:
                split = split_list.pop()
                proc, file = self._run_split(split)
                proc_split_file_dict[proc] = split, file

        return self.split_stat_dict

    def _run_split(self, split):
        # compute t2
        t2 = self.get_t2(split)

        # make it a z score
        t = np.sqrt(t2)

        # assume that covariance matrix is known (not estimated)
        z = t

        # https://github.com/spisakt/pTFCE/issues/3
        z[z > self.Z_MAX] = self.Z_MAX

        # write to file
        img_z = nib.Nifti1Image(z, self.file_tree.ref.affine)
        f_z = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
        img_z.to_filename(f_z)

        # compute smoothness via FSL
        smooth_dict = compute_smooth(f_z=f_z, f_mask=self.f_mask)
        v = int(smooth_dict['VOLUME'])
        r = smooth_dict['RESELS']

        # apply ptfce
        f_z_ptfce = tempfile.NamedTemporaryFile().name
        cmd = f'Rscript {f_run_ptfce} -i {f_z} -m {self.f_mask} -o {f_z_ptfce} -v {v} -r {r}'
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL)

        # todo: file ending always added in run_ptfce.R, even if already there
        f_z_ptfce += '.nii.gz'
        return proc, f_z_ptfce

    def run_split(self, split, **kwargs):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        proc, f_t2_ptfce = self._run_split(split)
        proc.wait()
        proc.kill()

        # read in file
        t2_ptfce = nib.load(f_t2_ptfce).get_data()

        return t2_ptfce
