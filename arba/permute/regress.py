import pathlib
import tempfile
from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import dice
from tqdm import tqdm

import arba
from mh_pytools import parallel
from ..region import RegionRegress
from ..seg_graph import SegGraphHistory


def get_r2(reg, **kwargs):
    return reg.r2


class PermuteRegress:
    """ runs permutation testing to find regions whose r^2 > 0 is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        num_perm (int): number of permutations to run
        alpha (float): confidence threshold
        feat_sbj (np.array): (todo x todo) subject features
        file_tree (FileTree): observed imaging data
    """

    def __init__(self, feat_sbj, file_tree, alpha=.05, num_perm=100,
                 target_mask=None, verbose=True, folder=None, par_flag=False):
        self.alpha = alpha
        self.num_perm = num_perm
        self.feat_sbj = feat_sbj
        self.file_tree = file_tree
        self.target_mask = target_mask
        self.verbose = verbose

        self.sg_hist = None
        self.merge_record = None

        self.folder = folder
        if self.folder is None:
            self.folder = pathlib.Path(tempfile.mkdtemp())

        with file_tree.loaded():
            self.run_single()

            self.permute(par_flag=par_flag)

            # compute pval + z score per node
            self.node_pval_dict = dict()
            self.node_z_dict = dict()
            self.node_r2_dict = self.merge_record.fnc_node_val_list['r2']
            for n in self.merge_record.nodes:
                reg_size = self.merge_record.node_size_dict[n]
                r2 = self.node_r2_dict[n]

                self.node_pval_dict[n] = self.get_pval(r2, reg_size)
                self.node_z_dict[n] = self.get_r2_z_score(r2, reg_size)

            node_p_negz_dict = {n: (p, -self.node_z_dict[n])
                                for n, p in self.node_pval_dict.items() if
                                p <= self.alpha}
            self.sig_node_cover = \
                self.merge_record._cut_greedy(node_p_negz_dict, max_flag=False)

            self.save()

    def run_single(self, _seed=None):
        """ runs a single Agglomerative Clustering run

        Args:
            _seed (int): if passed, toggles 'permutation testing' mode.
                         shuffles assignment of subject features to imaging
                         features and returns sg_hist without running
                         agglomerative clustering
        """
        if _seed is not None:
            RegionRegress.shuffle_feat_sbj(seed=_seed)
        else:
            RegionRegress.set_feat_sbj(feat_sbj=self.feat_sbj,
                                       sbj_list=self.file_tree.sbj_list)

        sg_hist = SegGraphHistory(file_tree=self.file_tree,
                                  cls_reg=arba.region.RegionRegress,
                                  fnc_dict={'r2': get_r2})

        if _seed is not None:
            return sg_hist

        self.sg_hist = sg_hist
        self.sg_hist.reduce_to(1, verbose=self.verbose)

        self.merge_record = self.sg_hist.merge_record

    def run_single_permute(self, **kwargs):
        sg_hist = self.run_single(**kwargs)

        merge_record = sg_hist.merge_record
        r2_list = merge_record.fnc_node_val_list['r2'].values()

        return sorted(r2_list, reverse=True)

    def permute(self, par_flag=False):
        # if seed = 0, evaluates as false and doesn't do anything
        seed_list = np.arange(1, self.num_perm + 1)
        arg_list = [{'_seed': x} for x in seed_list]

        if par_flag:
            val_list = parallel.run_par_fnc(obj=self, fnc='run_single_permute',
                                            arg_list=arg_list,
                                            verbose=self.verbose)
        else:
            val_list = list()
            for d in tqdm(arg_list, desc='permute', disable=not self.verbose):
                val_list.append(self.run_single_permute(**d))

        self._compute_r2_bounds(val_list)

        # ensure that RegionRegress has appropriate feat_sbj ordering
        RegionRegress.set_feat_sbj(feat_sbj=self.feat_sbj,
                                   sbj_list=self.file_tree.sbj_list)

        return val_list

    def get_pval(self, r2, reg_size):
        _r2_null = self.r2_null[:, reg_size - 1]
        return 1 - bisect_right(_r2_null, r2) / self.num_perm

    def get_r2_z_score(self, r2, reg_size):
        mu = self.mu_null[reg_size - 1]
        std = self.std_null[reg_size - 1]
        return (r2 - mu) / std

    def _compute_r2_bounds(self, val_list):
        """ computes bound on r2, per region size, for every permutation """
        # build r2_null, has shape (num_perm, num_vox).  each column is a
        # sorted list of upper bounds on r2 per permutation.
        self.r2_null = np.vstack(val_list)
        num_perm, num_vox = self.r2_null.shape

        # sum and normalize
        r2_null = np.cumsum(self.r2_null, axis=1)
        r2_null = r2_null / np.arange(1, num_vox + 1)

        # sort per region size
        r2_null = np.sort(r2_null, axis=0)

        # compute stats (for z)
        self.mu_null = np.mean(r2_null, axis=0)
        self.std_null = np.std(r2_null, axis=0)

    def save(self, size_v_r2=True, size_v_r2_pval=True, size_v_r2_z=True,
             size_v_r2_null=True, mask_detected=True, print_node=True):

        if size_v_r2:
            self.merge_record.plot_size_v('r2', label='r2',
                                          mask=self.target_mask,
                                          log_y=False)
            arba.plot.save_fig(self.folder / 'size_v_r2.pdf')

        if size_v_r2_pval:
            self.merge_record.plot_size_v(self.node_pval_dict, label='pval',
                                          mask=self.target_mask,
                                          log_y=False)
            arba.plot.save_fig(self.folder / 'size_v_pval.pdf')

        if size_v_r2_z:
            self.merge_record.plot_size_v(self.node_z_dict, label='r2z',
                                          mask=self.target_mask,
                                          log_y=False)
            arba.plot.save_fig(self.folder / 'size_v_r2z_score.pdf')

        if mask_detected:
            mask_estimate = self.merge_record.build_mask(self.sig_node_cover)
            mask_estimate.to_nii(self.folder / 'mask_detected.nii.gz')

        if size_v_r2_null:
            # print percentiles of r2 per size
            sns.set(font_scale=1.2)
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
            arba.plot.save_fig(self.folder / 'size_v_r2_null.pdf')

        if self.target_mask is not None:
            compute_print_dice(mask_estimate=mask_estimate,
                               mask_target=self.target_mask,
                               save_folder=self.folder)

            f_mask = self.folder / 'target_mask.nii.gz'
            self.target_mask.to_nii(f_mask)

        if print_node:
            for n in self.sig_node_cover:
                r = self.merge_record.resolve_node(n,
                                                   file_tree=self.sg_hist.file_tree,
                                                   reg_cls=arba.region.RegionRegress)
                r.pc_ijk.to_mask().to_nii(self.folder / f'node_{n}.nii.gz')
                r.plot(img_feat_label='fa')
                arba.plot.save_fig(self.folder / f'node_{n}.pdf')


def compute_print_dice(mask_estimate, mask_target, save_folder):
    mask_estimate = mask_estimate.astype(bool)
    mask_target = mask_target.astype(bool)
    dice_score = 1 - dice(mask_estimate.flatten(), mask_target.flatten())
    with open(str(save_folder / 'dice.txt'), 'w') as f:
        print(f'dice is {dice_score:.3f}', file=f)
        print(f'target vox: {mask_target.sum()}', file=f)
        print(f'detected vox: {mask_estimate.sum()}', file=f)
        true_detect = (mask_target & mask_estimate).sum()
        print(f'true detected vox: {true_detect}', file=f)
        false_detect = (~mask_target & mask_estimate).sum()
        print(f'false detected vox: {false_detect}', file=f)

    return dice_score
