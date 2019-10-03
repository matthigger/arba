import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import dice

from .get_pval import get_pval
from .permute import Permute
from ..plot import save_fig
from ..region import RegionRegress
from ..seg_graph import SegGraphHistory


def get_r2(reg, **kwargs):
    return reg.r2


class PermuteRegress(Permute):
    """ runs permutation testing to find regions whose r^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        data_sbj (DataSubject): observed subject data
    """

    def __init__(self, data_sbj, *args, **kwargs):
        self.data_sbj = data_sbj

        super().__init__(*args, **kwargs)

        # get mask of estimate
        self.node_pval_dict = dict()
        self.node_z_dict = dict()
        self.node_r2_dict = self.merge_record.fnc_node_val_list['r2']
        self.sig_node = self.get_sig_node()

        # build estimate mask
        arba_mask = self.merge_record.build_mask(self.sig_node)
        self.mode_est_mask_dict = {'arba': arba_mask}

    def _set_seed(self, seed=None):
        self.data_sbj.permute(seed)
        RegionRegress.set_data_sbj(self.data_sbj)

    def get_sg_hist(self, seed=None):
        self._set_seed(seed)
        return SegGraphHistory(data_img=self.data_img,
                               cls_reg=RegionRegress,
                               fnc_dict={'r2': get_r2})

    def run_single_permute(self, seed):
        sg_hist = self.get_sg_hist(seed)
        r2_list = sg_hist.merge_record.fnc_node_val_list['r2'].values()

        return {'r2': sorted(r2_list, reverse=True)}

    def permute(self, par_flag=False):
        val_list = super().permute(par_flag=par_flag)

        self.r2_null = np.vstack(d['r2'] for d in val_list)
        num_perm, num_vox = self.r2_null.shape

        # sum and normalize
        self.r2_null = np.cumsum(self.r2_null, axis=1)
        self.r2_null = self.r2_null / np.arange(1, num_vox + 1)

        # sort per region size
        self.r2_null = np.sort(self.r2_null, axis=0)

        # compute stats (for z)
        self.mu_null = np.mean(self.r2_null, axis=0)
        self.std_null = np.std(self.r2_null, axis=0)

        return val_list

    def get_sig_node(self):
        # compute pval + z score per node
        for n in self.merge_record.nodes:
            reg_size = self.merge_record.node_size_dict[n]
            r2 = self.node_r2_dict[n]

            self.node_pval_dict[n] = \
                get_pval(stat=r2,
                         stat_null=self.r2_null[:, reg_size - 1],
                         sort_flag=False,
                         stat_include_flag=True)
            self.node_z_dict[n] = self.get_r2_z_score(r2, reg_size)

        node_p_negz_dict = {n: (p, -self.node_z_dict[n])
                            for n, p in self.node_pval_dict.items() if
                            p <= self.alpha}

        return self.merge_record._cut_greedy(node_p_negz_dict, max_flag=False)

    def get_r2_z_score(self, r2, reg_size):
        mu = self.mu_null[reg_size - 1]
        std = self.std_null[reg_size - 1]
        return (r2 - mu) / std

    def save(self, size_v_r2=True, size_v_r2_pval=True, size_v_r2_z=True,
             size_v_r2_null=True, print_node=True):

        self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        if size_v_r2:
            self.merge_record.plot_size_v('r2', label='r2',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_r2.pdf')

        if size_v_r2_pval:
            self.merge_record.plot_size_v(self.node_pval_dict, label='pval',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_pval.pdf')

        if size_v_r2_z:
            self.merge_record.plot_size_v(self.node_z_dict, label='r2z',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_r2z_score.pdf')

        for label, mask in self.mode_est_mask_dict.items():
            mask.to_nii(self.folder / f'mask_est_{label}.nii.gz')

        if size_v_r2_null:
            # print percentiles of r2 per size
            cmap = plt.get_cmap('viridis')
            size = np.arange(1, self.r2_null.shape[1] + 1)
            p_list = [50, 75, 90, 95, 99]
            for p_idx, p in enumerate(p_list):
                perc_line = np.percentile(self.r2_null, p, axis=0)
                plt.plot(size, perc_line, label=f'{p}-th percentile',
                         color=cmap(p_idx / len(p_list)))
            plt.ylabel('r2')
            plt.xlabel('size')
            plt.gca().set_xscale('log')
            plt.legend()
            save_fig(self.folder / 'size_v_r2_null.pdf')

        if self.mask_target is not None:
            f_mask = self.folder / 'mask_target.nii.gz'
            self.mask_target.to_nii(f_mask)

            for label, mask in self.mode_est_mask_dict.items():
                compute_print_dice(mask_estimate=mask,
                                   mask_target=self.mask_target,
                                   save_folder=self.folder,
                                   label=label)

        if print_node:
            for n in self.sig_node:
                r = self.merge_record.resolve_node(n,
                                                   data_img=self.sg_hist.data_img,
                                                   reg_cls=RegionRegress)
                r.pc_ijk.to_mask().to_nii(self.folder / f'node_{n}.nii.gz')
                r.plot(img_idx=0,
                       img_label=f'mean {self.data_img.feat_list[0]}',
                       sbj_idx=1,
                       sbj_label=self.data_sbj.feat_list[1])
                save_fig(self.folder / f'node_{n}.pdf')


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
