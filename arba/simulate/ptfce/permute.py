import os
import pathlib
import shlex
import subprocess
import tempfile

import nibabel as nib

from ...permute_base import PermuteBase

f_run_ptfce = pathlib.Path(__file__).parent / 'run_ptfce.R'


class PermutePTFCE(PermuteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # write mask to temp file
        self.f_mask = self.file_tree.mask.to_nii()

    def __del__(self):
        os.remove(str(self.f_mask))

    def run_split(self, split, **kwargs):
        """ returns a volume of tfce enhanced t2 stats

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this split

        Returns:
            stat_volume (np.array): (space0, space1, space2)
        """
        # compute t2
        t2 = self.get_t2(split)

        # write to file
        img_t2 = nib.Nifti1Image(t2, self.file_tree.ref.affine)
        f_t2 = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
        img_t2.to_filename(f_t2)

        # apply ptfce
        f_t2_ptfce = tempfile.NamedTemporaryFile().name
        cmd = f'Rscript {f_run_ptfce} -i {f_t2} -m {self.f_mask} -o {f_t2_ptfce}'
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL)
        p.wait()
        p.kill()

        # read in file
        # kludge: R's dcemriS4 writeNIfTI appends .nii.gz, even if it already
        # has this suffix
        f_t2_ptfce += '.nii.gz'
        t2_ptfce = nib.load(f_t2_ptfce).get_data()

        return t2_ptfce
