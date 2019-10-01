import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import dice

from .permute import Permute
from ..plot import save_fig
from ..region import RegionDiscriminate
from ..seg_graph import SegGraphHistory


def get_t2(reg, **kwargs):
    return reg.t2


class PermuteDiscriminate(Permute):
    """ runs permutation testing to find regions whose t^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        split (Split): keys are grps, values are list of sbj
    """

    def __init__(self, *args, split, **kwargs):
        self.split = split

        super().__init__(*args, **kwargs)

        # get mask of estimate
        self.node_t2_dict = self.merge_record.fnc_node_val_list['t2']
        self.sig_node = self.get_sig_node()

        # build estimate mask
        arba_mask = self.merge_record.build_mask(self.sig_node)
        self.mode_est_mask_dict = {'arba': arba_mask}

    def _set_seed(self, seed=None):
        split = self.split.shuffle(seed=seed)
        RegionDiscriminate.set_split(split)

    def get_sg_hist(self, seed=None):
        self._set_seed(seed)
        return SegGraphHistory(data_img=self.data_img,
                               cls_reg=RegionDiscriminate,
                               fnc_dict={'t2': get_t2})

    def run_single_permute(self, seed):
        sg_hist = self.get_sg_hist(seed)
        t2_list = sg_hist.merge_record.fnc_node_val_list['t2'].values()

        return {'t2': sorted(t2_list, reverse=True)}

    def permute(self, par_flag=False):
        val_list = super().permute(par_flag=par_flag)

        self.t2_null = np.vstack(d['t2'] for d in val_list)

        return val_list

    def get_sig_node(self):

        # get nodes with largest t2
        node_t2_dict = self.merge_record.fnc_node_val_list['t2']
        sig_node = self.merge_record._cut_greedy(node_t2_dict, max_flag=False)

        # choose only significant nodes
        thresh = np.percentile(self.t2_null[:, 0], (1 - self.alpha) * 100)
        sig_node = [n for n in sig_node if node_t2_dict[n] >= thresh]

        return sig_node

    def save(self, size_v_t2=True, size_v_t2_null=True):

        self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        if size_v_t2:
            self.merge_record.plot_size_v('t2', label='t2',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_t2.pdf')

        for label, mask in self.mode_est_mask_dict.items():
            mask.to_nii(self.folder / f'mask_est_{label}.nii.gz')

        if size_v_t2_null:
            # print percentiles of t2 per size
            cmap = plt.get_cmap('viridis')
            size = np.arange(1, self.t2_null.shape[1] + 1)
            p_list = [50, 75, 90, 95, 99]
            for p_idx, p in enumerate(p_list):
                perc_line = np.percentile(self.t2_null, p, axis=0)
                plt.plot(size, perc_line, label=f'{p}-th percentile',
                         color=cmap(p_idx / len(p_list)))
            plt.ylabel('t2')
            plt.xlabel('size')
            plt.gca().set_xscale('log')
            plt.legend()
            save_fig(self.folder / 'size_v_t2_null.pdf')

        if self.mask_target is not None:
            f_mask = self.folder / 'mask_target.nii.gz'
            self.mask_target.to_nii(f_mask)

            for label, mask in self.mode_est_mask_dict.items():
                compute_print_dice(mask_estimate=mask,
                                   mask_target=self.mask_target,
                                   save_folder=self.folder,
                                   label=label)


def compute_print_dice(mask_estimate, mask_target, save_folder, label=None):
    mask_estimate = mask_estimate.astype(bool)
    mask_target = mask_target.astype(bool)
    dice_score = 1 - dice(mask_estimate.flatten(), mask_target.flatten())
    with open(str(save_folder / 'dice.txt'), 'a+') as f:
        print(f'---{label}---', file=f)
        print(f'dice is {dice_score:.3f}', file=f)
        print(f'target vox: {mask_target.sum()}', file=f)
        print(f'detected vox: {mask_estimate.sum()}', file=f)
        true_detect = (mask_target & mask_estimate).sum()
        print(f'true detected vox: {true_detect}', file=f)
        false_detect = (~mask_target & mask_estimate).sum()
        print(f'false detected vox: {false_detect}', file=f)

    return dice_score
