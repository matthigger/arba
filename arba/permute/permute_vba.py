import nibabel as nib
import numpy as np

from .permute import Permute
from .permute_discrim import PermuteDiscriminate
from .permute_regress import PermuteRegress
from ..space import Mask
from ..tfce import apply_tfce


class PermuteVBA(Permute):
    """ runs vanilla VBA and TFCE alongside permutation testing

    Attributes:
        mode_stat_dict (dict): keys are vba subtype labels ('vba', 'tfce'),
                               values are arrays of stats in shape of image
        mode_null_dict (dict): keys are vba subtype labels, values are sorted
                               arrays of size num_perm, each containing max
                               stat across image per permutation
    """

    def __init__(self, *args, **kwargs):
        self.vba_stat_dict = dict()
        self.vba_null_dict = dict()

        super().__init__(*args, **kwargs)

        for mode, null in self.vba_null_dict.items():
            cutoff = np.percentile(null, 100 * (1 - self.alpha))
            mask = self.vba_stat_dict[mode] >= cutoff
            self.mode_est_mask_dict[mode] = Mask(mask, ref=self.data_img.ref)

    def run_single(self):
        super().run_single()

        self.vba_stat_dict['vba'] = self.merge_record.get_array(self.stat)
        self.vba_stat_dict['tfce'] = apply_tfce(self.vba_stat_dict['vba'])

    def permute(self, *args, **kwargs):
        val_list = super().permute(*args, **kwargs)

        self.vba_null_dict['vba'] = self.stat_null[:, 0]
        self.vba_null_dict['tfce'] = np.array([d['tfce'] for d in val_list])

        return val_list

    def run_single_permute(self, seed):
        # this method needs access to sg_hist, we build and pass to avoid
        # redundant computation
        sg_hist = self.get_sg_hist(seed)
        stat_dict = super().run_single_permute(_sg_hist=sg_hist)

        # get stat img and apply tfce
        stat = sg_hist.to_array(attr=self.stat)
        stat_tfce = apply_tfce(stat)

        # sort tfce values from largest to smallest
        stat_dict['tfce'] = max(stat_tfce[self.data_img.mask])

        return stat_dict

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        for label, img in self.vba_stat_dict.items():
            f = self.folder / f'{self.stat}_{label}.nii.gz'
            img = nib.Nifti1Image(img, affine=self.data_img.ref.affine)
            img.to_filename(str(f))


PermuteDiscriminateVBA = type('PermuteDiscriminateVBA',
                              (PermuteVBA,),
                              dict(PermuteDiscriminate.__dict__))

PermuteRegressVBA = type('PermuteRegressVBA',
                         (PermuteVBA,),
                         dict(PermuteRegress.__dict__))
