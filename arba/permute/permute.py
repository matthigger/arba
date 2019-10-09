import abc
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import dice
from tqdm import tqdm

from mh_pytools import parallel
from .get_pval import get_pval
from ..plot import save_fig
from ..seg_graph import SegGraphHistory


class Permute:
    """ runs permutation testing to find regions whose stat is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        num_perm (int): number of permutations to run
        alpha (float): confidence threshold
        data_img (DataImage): observed imaging data
    """

    def __init__(self, data_img, alpha=.05, num_perm=100, mask_target=None,
                 verbose=True, folder=None, par_flag=False, stat_save=tuple()):
        assert alpha >= 1 / (num_perm + 1), \
            'not enough perm for alpha, never sig'

        self.alpha = alpha
        self.num_perm = num_perm
        self.data_img = data_img
        self.mask_target = mask_target
        self.verbose = verbose

        self.stat_null = None
        self.node_pval_dict = dict()
        self.node_z_dict = dict()

        self.sg_hist = None
        self.merge_record = None

        self.stat_save = (*stat_save, self.stat)

        self.folder = folder
        if self.folder is None:
            self.folder = pathlib.Path(tempfile.mkdtemp())

        with data_img.loaded():
            self.run_single()

            self.permute(par_flag=par_flag)

        # get mask of estimate
        self.node_stat_dict = self.merge_record.stat_node_val_dict[self.stat]
        self.sig_node = self.get_sig_node()

        # build estimate mask
        arba_mask = self.merge_record.build_mask(self.sig_node)
        self.mode_est_mask_dict = {'arba': arba_mask}

    @abc.abstractmethod
    def _set_seed(self, seed=None):
        raise NotImplementedError

    def get_sg_hist(self, seed=None):
        self._set_seed(seed)
        return SegGraphHistory(data_img=self.data_img,
                               cls_reg=self.reg_cls,
                               stat_save=self.stat_save)

    def run_single_permute(self, seed=None, _sg_hist=None):
        if _sg_hist is None:
            assert seed is not None, 'seed required'
            sg_hist = self.get_sg_hist(seed)
        else:
            sg_hist = _sg_hist

        sg_hist.reduce_to(1, verbose=self.verbose)

        merge_record = sg_hist.merge_record
        max_size = len(merge_record.leaf_ijk_dict)

        # record max stat per size
        node_stat_dict = merge_record.stat_node_val_dict[self.stat]
        max_stat = np.zeros(max_size)
        for node, stat in node_stat_dict.items():
            size = merge_record.node_size_dict[node]
            if stat > max_stat[size - 1]:
                max_stat[size - 1] = stat

        # record which sizes are observed (unobserved will be interpolated)
        observed = max_stat != 0

        # ensure monotonicity (increase max_stat values)
        for idx in range(len(max_stat) - 2, 0, -1):
            if max_stat[idx] < max_stat[idx + 1]:
                max_stat[idx] = max_stat[idx + 1]

        # fill unobserved with linearly interpolated values
        size = np.array(range(max_size))
        fnc = interp1d(size[observed], max_stat[observed])
        max_stat = fnc(size)

        return {self.stat: max_stat}

    def run_single(self):
        """ runs a single Agglomerative Clustering run
        """
        # build sg_hist, reduce
        self.sg_hist = self.get_sg_hist()
        self.sg_hist.reduce_to(1, verbose=self.verbose)
        self.merge_record = self.sg_hist.merge_record

    def permute(self, par_flag=False):
        # if seed = 0, evaluates as false and doesn't do anything
        seed_list = np.arange(1, self.num_perm + 1)
        arg_list = [{'seed': x} for x in seed_list]

        if par_flag:
            val_list = parallel.run_par_fnc(obj=self, fnc='run_single_permute',
                                            arg_list=arg_list,
                                            verbose=self.verbose)
        else:
            val_list = list()
            for d in tqdm(arg_list, desc='permute', disable=not self.verbose):
                val_list.append(self.run_single_permute(**d))

        # reset permutation to original data
        self._set_seed(seed=None)

        # record null stats
        self.stat_null = np.vstack(d[self.stat] for d in val_list)
        self.stat_null = np.sort(self.stat_null, axis=0)

        return val_list

    def get_sig_node(self):
        # get nodes with largest stat
        self.node_pval_dict = dict()
        self.node_z_dict = dict()
        for n in self.merge_record.nodes:
            reg_size = self.merge_record.node_size_dict[n]
            stat = self.node_stat_dict[n]
            stat_null = self.stat_null[:, reg_size - 1]

            self.node_pval_dict[n] = get_pval(stat=stat,
                                              stat_null=stat_null,
                                              sort_flag=False,
                                              stat_include_flag=True)

            mu, std = np.mean(stat_null), np.std(stat_null)
            self.node_z_dict[n] = (stat - mu) / std

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

    def save(self, size_v_stat=True, size_v_stat_null=True,
             size_v_stat_pval=True, print_node=True, size_v_stat_z=True):

        self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        if size_v_stat:
            self.merge_record.plot_size_v(self.stat, label=self.stat,
                                          mask=self.mask_target,
                                          log_y=True)

            # get mu, std per size
            max_size = max(self.merge_record.node_size_dict.values())
            size = range(1, max_size + 1)
            p = (1 - self.alpha) * 100
            max_stat = np.percentile(self.stat_null, p,
                                     axis=0)
            plt.plot(size, max_stat, linestyle='-', color='g', linewidth=3,
                     label=f'{p:.0f}%')
            plt.legend()

            save_fig(self.folder / f'size_v_{self.stat}.pdf')

        for label, mask in self.mode_est_mask_dict.items():
            mask.to_nii(self.folder / f'mask_est_{label}.nii.gz')

        if size_v_stat_pval:
            self.merge_record.plot_size_v(self.node_pval_dict, label='pval',
                                          mask=self.mask_target,
                                          log_y=True)
            save_fig(self.folder / f'size_v_{self.stat}pval.pdf')

        if size_v_stat_z:
            self.merge_record.plot_size_v(self.node_z_dict,
                                          label=f'{self.stat}z',
                                          mask=self.mask_target,
                                          log_y=False)
            save_fig(self.folder / f'size_v_{self.stat}_z_score.pdf')

        if size_v_stat_null:
            # print percentiles of stat per size
            cmap = plt.get_cmap('viridis')
            size = np.arange(1, self.stat_null.shape[1] + 1)
            p_list = [50, 75, 90, 95, 99]
            for p_idx, p in enumerate(p_list):
                perc_line = np.percentile(self.stat_null, p, axis=0)
                plt.plot(size, perc_line, label=f'{p}-th percentile',
                         color=cmap(p_idx / len(p_list)))
            plt.ylabel(self.stat)
            plt.xlabel('size')
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            plt.legend()
            save_fig(self.folder / f'size_v_{self.stat}_null.pdf')

        if self.mask_target is not None:
            f_mask = self.folder / 'mask_target.nii.gz'
            self.mask_target.to_nii(f_mask)

            for label, mask in self.mode_est_mask_dict.items():
                compute_print_dice(mask_estimate=mask,
                                   mask_target=self.mask_target,
                                   save_folder=self.folder,
                                   label=label)

        if print_node and hasattr(self.reg_cls, 'plot'):
            for n in self.sig_node:
                r = self.merge_record.resolve_node(n,
                                                   data_img=self.sg_hist.data_img,
                                                   reg_cls=self.reg_cls)
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
