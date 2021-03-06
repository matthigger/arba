import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns

from .permute_discrim import PermuteDiscriminate
from .permute_regress import PermuteRegress
from ..plot import save_fig
from ..space import Mask
from ..tfce import apply_tfce


# todo: avoid redundant code but keep PermuteXXXVBA pickle-able (multiprocess)

class PermuteDiscriminateVBA(PermuteDiscriminate):
    """ runs vanilla VBA and TFCE alongside permutation testing

    Attributes:
        vba_stat_dict (dict): keys are vba subtype labels ('vba', 'tfce'),
                              values are arrays of stats in shape of image
        vba_null_dict (dict): keys are vba subtype labels, values are sorted
                              arrays of size num_perm, each containing max
                              stat across image per permutation
        vba_thresh_dict (dict): keys are vba subtype labels, values are sig
                                thresholds
    """

    def __init__(self, *args, **kwargs):
        self.vba_stat_dict = dict()
        self.vba_null_dict = dict()
        self.vba_thresh_dict = dict()

        super().__init__(*args, **kwargs)

        for mode, null in self.vba_null_dict.items():
            mask = self.vba_stat_dict[mode] >= self.vba_thresh_dict[mode]
            self.mode_est_mask_dict[mode] = Mask(mask, ref=self.data_img.ref)

    def run_single(self):
        super().run_single()

        self.vba_stat_dict['vba'] = self.merge_record.get_array(self.stat)
        self.vba_stat_dict['tfce'] = apply_tfce(self.vba_stat_dict['vba'])

    def permute_samples(self, *args, **kwargs):
        val_list = super().permute_samples(*args, **kwargs)

        self.vba_null_dict['vba'] = np.array([d['vba'] for d in val_list])
        self.vba_null_dict['tfce'] = np.array([d['tfce'] for d in val_list])

        # compute signifigance thresholds
        perc = 100 * (1 - self.alpha)
        for mode, null in self.vba_null_dict.items():
            self.vba_thresh_dict[mode] = np.percentile(null, perc)

        return val_list

    def run_single_permute(self, seed):
        # get stat img and apply tfce
        sg_hist = self.get_sg_hist(seed)
        stat = sg_hist.to_array(attr=self.stat)
        stat_tfce = apply_tfce(stat)

        # this method needs access to sg_hist, we build and pass to avoid
        # redundant computation
        stat_dict = super().run_single_permute(_sg_hist=sg_hist)

        # save max stat
        stat_dict['vba'] = max(stat[self.data_img.mask])
        stat_dict['tfce'] = max(stat_tfce[self.data_img.mask])

        return stat_dict

    def save(self, *args, null_vba=False, **kwargs):
        super().save(*args, **kwargs)

        for label, img in self.vba_stat_dict.items():
            f = self.folder / f'{self.stat}_{label}.nii.gz'
            img = nib.Nifti1Image(img, affine=self.data_img.ref.affine)
            img.to_filename(str(f))

        if null_vba:
            for label, stat in self.vba_null_dict.items():
                sns.set()
                plt.hist(stat, bins=25)
                plt.axvline(self.vba_thresh_dict[label], label='sig thresh',
                            color='g', linewidth=3)
                plt.legend()
                plt.xlabel(label)
                plt.ylabel('count')
                save_fig(self.folder / f'vba_hist_{label}.pdf')


class PermuteRegressVBA(PermuteRegress):
    """ runs vanilla VBA and TFCE alongside permutation testing

    Attributes:
        vba_stat_dict (dict): keys are vba subtype labels ('vba', 'tfce'),
                              values are arrays of stats in shape of image
        vba_null_dict (dict): keys are vba subtype labels, values are sorted
                              arrays of size num_perm, each containing max
                              stat across image per permutation
        vba_thresh_dict (dict): keys are vba subtype labels, values are sig
                                thresholds
    """

    def __init__(self, *args, **kwargs):
        self.vba_stat_dict = dict()
        self.vba_null_dict = dict()
        self.vba_thresh_dict = dict()

        super().__init__(*args, **kwargs)

        for mode, null in self.vba_null_dict.items():
            mask = self.vba_stat_dict[mode] >= self.vba_thresh_dict[mode]
            self.mode_est_mask_dict[mode] = Mask(mask, ref=self.data_img.ref)

    def run_single(self):
        super().run_single()

        self.vba_stat_dict['vba'] = self.merge_record.get_array(self.stat)
        self.vba_stat_dict['tfce'] = apply_tfce(self.vba_stat_dict['vba'])

    def permute_samples(self, *args, **kwargs):
        val_list = super().permute_samples(*args, **kwargs)

        self.vba_null_dict['vba'] = np.array([d['vba'] for d in val_list])
        self.vba_null_dict['tfce'] = np.array([d['tfce'] for d in val_list])

        # compute signifigance thresholds
        perc = 100 * (1 - self.alpha)
        for mode, null in self.vba_null_dict.items():
            self.vba_thresh_dict[mode] = np.percentile(null, perc)

        return val_list

    def run_single_permute(self, seed):
        # get stat img and apply tfce
        sg_hist = self.get_sg_hist(seed)
        stat = sg_hist.to_array(attr=self.stat)
        stat_tfce = apply_tfce(stat)

        # this method needs access to sg_hist, we build and pass to avoid
        # redundant computation
        stat_dict = super().run_single_permute(_sg_hist=sg_hist)

        # save max stat
        stat_dict['vba'] = max(stat[self.data_img.mask])
        stat_dict['tfce'] = max(stat_tfce[self.data_img.mask])

        return stat_dict

    def save(self, *args, null_vba=False, **kwargs):
        super().save(*args, **kwargs)

        for label, img in self.vba_stat_dict.items():
            f = self.folder / f'{self.stat}_{label}.nii.gz'
            img = nib.Nifti1Image(img, affine=self.data_img.ref.affine)
            img.to_filename(str(f))

        if null_vba:
            for label, stat in self.vba_null_dict.items():
                sns.set()
                plt.hist(stat, bins=25)
                plt.axvline(self.vba_thresh_dict[label], label='sig thresh',
                            color='g', linewidth=3)
                plt.legend()
                plt.xlabel(label)
                plt.ylabel('count')
                save_fig(self.folder / f'vba_hist_{label}.pdf')
