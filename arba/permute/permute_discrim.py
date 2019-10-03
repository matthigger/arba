import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import dice

from .get_pval import get_pval
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

        self.node_pval_dict = dict()
        self.node_z_dict = dict()

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
        # run agglomerative clustering
        sg_hist = self.get_sg_hist(seed)
        sg_hist.reduce_to(1, verbose=self.verbose)

        merge_record = sg_hist.merge_record
        max_size = len(merge_record.leaf_ijk_dict)

        # record max t2 per size
        node_t2_dict = merge_record.fnc_node_val_list['t2']
        max_t2 = np.zeros(max_size)
        for node, t2 in node_t2_dict.items():
            size = merge_record.node_size_dict[node]
            if t2 > max_t2[size - 1]:
                max_t2[size - 1] = t2

        # record which sizes are observed (unobserved will be interpolated)
        observed = max_t2 != 0

        # ensure monotonicity (increase max_t2 values)
        for idx in range(len(max_t2) - 2, 0, -1):
            if max_t2[idx] < max_t2[idx + 1]:
                max_t2[idx] = max_t2[idx + 1]

        # fill unobserved with linearly interpolated values
        size = np.array(range(max_size))
        fnc = interp1d(size[observed], max_t2[observed])
        max_t2 = fnc(size)

        return {'t2': max_t2}

    def permute(self, par_flag=False):
        val_list = super().permute(par_flag=par_flag)

        self.t2_null = np.vstack(d['t2'] for d in val_list)
        self.t2_null = np.sort(self.t2_null, axis=0)

        return val_list

    def get_sig_node(self):
        # get nodes with largest t2
        node_t2_dict = self.merge_record.fnc_node_val_list['t2']
        self.node_pval_dict = dict()
        self.node_z_dict = dict()
        for n in self.merge_record.nodes:
            reg_size = self.merge_record.node_size_dict[n]
            t2 = node_t2_dict[n]
            stat_null = self.t2_null[:, reg_size - 1]

            self.node_pval_dict[n] = get_pval(stat=t2,
                                              stat_null=stat_null,
                                              sort_flag=False,
                                              stat_include_flag=True)

            self.node_z_dict[n] = (t2 - np.mean(stat_null)) / np.std(stat_null)

        # keys are nodes, values are tuples of pval and negative z score. these
        # tuples sort the nodes from most compelling (smallest value) to least
        node_p_negz_dict = dict()
        for n, p in self.node_pval_dict.items():
            if p > self.alpha:
                # node isn't significant
                continue
            node_p_negz_dict[n] = p, -self.node_z_dict[n]

        # cut to a disjoint set of the most compelling nodes
        sig_node = self.merge_record._cut_greedy(node_p_negz_dict,
                                                 max_flag=False)

        return sig_node

    def save(self, size_v_t2=True, size_v_t2_null=True, size_v_t2_pval=True,
             size_v_t2_z=True):

        self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        if size_v_t2:
            self.merge_record.plot_size_v('t2', label='t2',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_t2.pdf')

        for label, mask in self.mode_est_mask_dict.items():
            mask.to_nii(self.folder / f'mask_est_{label}.nii.gz')

        if size_v_t2_pval:
            self.merge_record.plot_size_v(self.node_pval_dict, label='pval',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_pval.pdf')

        if size_v_t2_z:
            self.merge_record.plot_size_v(self.node_z_dict, label='r2z',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / 'size_v_r2z_score.pdf')

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
