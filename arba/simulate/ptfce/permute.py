import multiprocessing
import os
import pathlib
import shlex
import subprocess
import tempfile
from time import sleep

import nibabel as nib
import psutil

from ...permute_base import PermuteBase

f_run_ptfce = pathlib.Path(__file__).parent / 'run_ptfce.R'


class PermutePTFCE(PermuteBase):
    # memory needed to run each
    mem_buffer = 8e9

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

        proc_split_file_dict = dict()
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

            # remove finished processes
            for proc in proc_to_rm:
                del proc_split_file_dict[proc]

            # no more memory, dont start new job
            if psutil.virtual_memory().available < self.mem_buffer:
                continue

            # no more cpu, dont start new job
            if len(proc_split_file_dict) > multiprocessing.cpu_count():
                continue

            # start new job
            split = split_list.pop()
            proc, file = self._run_split(split)
            proc_split_file_dict[proc] = split, file

        return self.split_stat_dict

    def _run_split(self, split):
        # compute t2
        t2 = self.get_t2(split)

        # write to file
        img_t2 = nib.Nifti1Image(t2, self.file_tree.ref.affine)
        f_t2 = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
        img_t2.to_filename(f_t2)

        # apply ptfce
        f_t2_ptfce = tempfile.NamedTemporaryFile().name
        cmd = f'Rscript {f_run_ptfce} -i {f_t2} -m {self.f_mask} -o {f_t2_ptfce}'
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL)
        return proc, f_t2_ptfce

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
        # kludge: R's dcemriS4 writeNIfTI appends .nii.gz, even if it already
        # has this suffix
        f_t2_ptfce += '.nii.gz'
        t2_ptfce = nib.load(f_t2_ptfce).get_data()

        return t2_ptfce
