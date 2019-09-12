import nibabel as nib
import numpy as np
from tqdm import tqdm

from mh_pytools import parallel
from .permute_regress import PermuteRegress, get_r2, compute_print_dice
from ..region import RegionRegress
from ..seg_graph import SegGraph
from ..tfce import apply_tfce


class PermuteRegressVBA(PermuteRegress):
    """ runs vanilla VBA, TFCE and pTFCE variants of regression permutation

    Attributes:
        vba_r2_dict (dict): keys are vba subtype labels ('vba', 'tfce'), values
                            are arrays of r2 values
        vba_r2_null_dict (dict): keys are vba subtype labels, values are max r2
                                 across all voxels within a permutation
        vba_mask_estimate_dict (dict): keys are vba subtype labels, values are
                                       masks of detected areas
    """

    def __init__(self, *args, par_flag=False, save_flag=True, **kwargs):
        super().__init__(*args, par_flag=par_flag, save_flag=save_flag,
                         **kwargs)

        self.vba_r2_dict = dict()
        self.vba_r2_null_dict = dict()
        self.vba_mask_estimate_dict = dict()

        self.run_single_vba()
        self.permute_vba(par_flag=par_flag)

        for vba, r2_null in self.vba_r2_null_dict.items():
            cutoff_r2 = np.percentile(r2_null, 100 * (1 - self.alpha))
            mask_estimate = self.vba_r2_dict[vba] >= cutoff_r2
            self.vba_mask_estimate_dict[vba] = mask_estimate

        if save_flag:
            self.save_vba()

    def run_single_vba(self, _seed=None):
        self.feat_sbj.permute(_seed)
        RegionRegress.set_sbj_feat(self.feat_sbj)

        sg = SegGraph(file_tree=self.file_tree,
                      cls_reg=RegionRegress)

        # get r2 img
        r2 = sg.to_array(fnc=get_r2)

        # get t2_tfce img
        r2_tfce = apply_tfce(r2)

        if _seed is not None:
            return max(r2[self.file_tree.mask]), \
                   max(r2_tfce[self.file_tree.mask])

        self.vba_r2_dict['vba'] = r2
        self.vba_r2_dict['tfce'] = r2_tfce

    def permute_vba(self, par_flag=False):
        # if seed = 0, evaluates as false and doesn't do anything
        seed_list = np.arange(1, self.num_perm + 1)
        arg_list = [{'_seed': x} for x in seed_list]

        if par_flag:
            val_list = parallel.run_par_fnc(obj=self,
                                            fnc='run_single_vba',
                                            arg_list=arg_list,
                                            verbose=self.verbose)
        else:
            val_list = list()
            for d in tqdm(arg_list, desc='permute',
                          disable=not self.verbose):
                val_list.append(self.run_single_permute(**d))

        # add in the unpermuted data
        val_list.append((self.vba_r2_dict['vba'].max(),
                         self.vba_r2_dict['tfce'].max()))

        self.vba_r2_null_dict['vba'] = sorted(x[0] for x in val_list)
        self.vba_r2_null_dict['tfce'] = sorted(x[1] for x in val_list)

        return val_list

    def save_vba(self, *args, **kwargs):
        for vba, mask_estimate in self.vba_mask_estimate_dict.items():
            img = nib.Nifti1Image(mask_estimate.astype(np.uint8),
                                  affine=self.file_tree.ref.affine)
            img.to_filename(str(self.folder / f'mask_estimate_{vba}.nii.gz'))

        if self.mask_target is not None:
            for vba, mask_estimate in self.vba_mask_estimate_dict.items():
                compute_print_dice(mask_estimate=mask_estimate,
                                   mask_target=self.mask_target,
                                   save_folder=self.folder,
                                   label=vba)
