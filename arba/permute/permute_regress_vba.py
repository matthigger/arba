import nibabel as nib
import numpy as np

from .permute_regress import PermuteRegress, get_r2
from ..space import Mask
from ..tfce import apply_tfce


class PermuteRegressVBA(PermuteRegress):
    """ runs vanilla VBA, TFCE and pTFCE variants of regression permutation

    Attributes:
        vba_r2_dict (dict): keys are vba subtype labels ('vba', 'tfce'), values
                            are arrays of r2 values
        vba_r2_null_dict (dict): keys are vba subtype labels, values are max r2
                                 across all voxels within a permutation
    """

    def __init__(self, *args, par_flag=False, **kwargs):
        self.vba_r2_dict = dict()
        self.vba_r2_null_dict = dict()
        self.mode_null_dict = dict()
        self.mode_est_mask_dict = dict()

        super().__init__(*args, par_flag=par_flag, **kwargs)

        for mode, null in self.mode_null_dict.items():
            cutoff_r2 = np.percentile(null, 100 * (1 - self.alpha))
            mask = self.vba_r2_dict[mode] >= cutoff_r2
            self.mode_est_mask_dict[mode] = Mask(mask, ref=self.data_img.ref)

    def run_single(self):
        super().run_single()

        self.vba_r2_dict['vba'] = self.merge_record.get_array('r2')
        self.vba_r2_dict['tfce'] = apply_tfce(self.vba_r2_dict['vba'])

    def permute(self, *args, **kwargs):
        val_list = super().permute(*args, **kwargs)

        self.mode_null_dict['vba'] = self.r2_null[:, 0]
        self.mode_null_dict['tfce'] = np.array([d['r2_tfce'] for d in val_list])

        return val_list

    def run_single_permute(self, seed=None):
        sg_hist = self.get_sg_hist(seed)
        r2_list = sg_hist.merge_record.fnc_node_val_list['r2'].values()

        stat_dict = {'r2': sorted(r2_list, reverse=True)}

        # get r2 img and apply tfce
        r2 = sg_hist.to_array(fnc=get_r2)
        r2_tfce = apply_tfce(r2)

        # sort tfce values from largest to smallest
        stat_dict['r2_tfce'] = max(r2_tfce[self.data_img.mask])

        return stat_dict

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        for label, img in self.vba_r2_dict.items():
            f = self.folder / f'r2_{label}.nii.gz'
            img = nib.Nifti1Image(img, affine=self.data_img.ref.affine)
            img.to_filename(str(f))
