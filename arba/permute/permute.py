import abc
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.spatial.distance import dice
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from mh_pytools import parallel
from ..plot import save_fig
from ..seg_graph import SegGraphHistory


class Permute:
    """ runs permutation testing to find regions whose stat is significant

    additionally, this object serves as a container for the result objects

    `model' refers to a distribution of how the statistic changes with region
    size.  it is estimated via bootstrapping from the permutation dataset, see
    permute_model()

    Attributes:
        num_perm (int): number of permutations to run
        alpha (float): confidence threshold
        data_img (DataImage): observed imaging data
    """

    def __init__(self, data_img, alpha=.05, num_perm=100, num_perm_model=10,
                 mask_target=None, verbose=True, folder=None, par_flag=False,
                 stat_save=tuple()):
        assert alpha >= 1 / (num_perm + 1), \
            'not enough perm for alpha, never sig'

        self.alpha = alpha
        self.num_perm = num_perm
        self.num_perm_model = num_perm_model
        self.data_img = data_img
        self.mask_target = mask_target
        self.verbose = verbose

        self.null_perc = None
        self.model_mu_lin_reg = None
        self.model_err_lin_reg = None

        self.sg_hist = None
        self.merge_record = None

        self.stat_save = (*stat_save, self.stat)

        self.folder = folder
        if self.folder is None:
            self.folder = pathlib.Path(tempfile.mkdtemp())

        with data_img.loaded():
            self.run_single()
            self.permute_model(par_flag=par_flag)
            self.permute_samples(par_flag=par_flag)

        # get mask of estimate
        self.node_stat_dict = self.merge_record.stat_node_val_dict[self.stat]
        self.sig_node = self.get_sig_node()

        # build estimate mask
        arba_mask = self.merge_record.build_mask(self.sig_node)
        self.mode_est_mask_dict = {'arba': arba_mask}

    def permute_model(self, *args, **kwargs):
        """ estimates distribution of how statistic varies with size

        this is achieved by bootstrap sampling stats from the null distribution
        via permutation testing.  once the stats are collected, we assume that

        log stat = a(size) + b(size) log size + err

        for err ~ N(0, sig(size))

        for linear functions a, b and sig
        """
        # draw samples from permutation testing
        val_list = self.permute(*args, **kwargs)

        # estimate mean stat per size
        size = np.vstack(d['size'] for d in val_list)
        stat = np.vstack(d['stat'] for d in val_list)
        self.model_mu_lin_reg = LinearRegression()
        self.model_mu_lin_reg.fit(size, stat)

        # compute error per size given model above
        err = stat - self.model_mu_lin_reg.predict(size)

        # estimate error bandwidth given model above
        self.model_err_lin_reg = LinearRegression()
        self.model_err_lin_reg.fit(size, np.abs(err))

    def get_percentile(self, stat, size):
        """ computes percentile of stat under model
        """
        stat = np.atleast_2d(stat).T
        size = np.atleast_2d(size).T

        diff = stat - self.model_mu_lin_reg.predict(size)
        z = np.divide(diff, self.model_err_lin_reg.predict(size))

        return scipy.stats.norm.cdf(z)

    def permute_samples(self, *args, **kwargs):
        """ draws samples from null distribution, saves max percentile
        """
        # draw samples from permutation testing
        val_list = self.permute(*args, **kwargs)

        # comptue percentile and store
        size = np.vstack(d['size'] for d in val_list)
        stat = np.vstack(d['stat'] for d in val_list)
        self.null_perc = self.get_percentile(stat, size)

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
        size = np.empty((len(merge_record.nodes, 1)))
        stat = np.empty((len(merge_record.nodes, 1)))
        for node_idx, node in enumerate(merge_record.nodes):
            size = merge_record.node_size_dict[node]
            stat = merge_record.stat_node_val_dict[self.stat][node]

        return {'size': size, 'stat': stat}

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

        return val_list

    def get_sig_node(self):
        # get size and stat
        size = np.empty((len(self.merge_record, 1)))
        stat = np.empty((len(self.merge_record, 1)))
        for n in self.merge_record.nodes:
            size[n] = self.merge_record.node_size_dict[n]
            stat[n] = self.merge_record.stat_node_val_dict[self.stat][n]
        perc = self.get_percentile(stat, size)
        node_perc_dict = dict(enumerate(perc))

        # cut to a disjoint set of the most compelling nodes (max perc)
        node_best = self.merge_record._cut_greedy(node_perc_dict,
                                                 max_flag=True)

        # get only the significant nodes
        thresh = np.percentile(self.null_perc, 100 * (1 - self.alpha))
        sig_node = {n for n in node_best if node_perc_dict[n] >= thresh}

        return sig_node

    def save(self, size_v_stat=True, size_v_stat_null=True,
             size_v_stat_pval=True, print_node=True, performance=True):

        def plot_thresh(p_list):
            size = np.arange(1, self.stat_null.shape[1] + 1)
            plt.plot(size, 10 ** self.log_stat_predict, label='null mean')
            for idx, p in enumerate(p_list):
                stat = self.log_stat_predict + \
                       self.log_stat_err_std * scipy.stats.norm.isf(p)
                plt.plot(size, 10 ** stat, label=f'null p={p}')

        self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        if size_v_stat:
            self.merge_record.plot_size_v(self.stat, label=self.stat,
                                          mask=self.mask_target,
                                          log_y=True)
            plot_thresh(p_list=[.05, .01, .001])
            plt.legend()

            save_fig(self.folder / f'size_v_{self.stat}.pdf')

        for label, mask in self.mode_est_mask_dict.items():
            mask.to_nii(self.folder / f'mask_est_{label}.nii.gz')

        if size_v_stat_pval:
            self.merge_record.plot_size_v(self.node_pval_dict, label='pval',
                                          mask=self.mask_target,
                                          log_y=True)
            save_fig(self.folder / f'size_v_{self.stat}pval.pdf')

        if size_v_stat_null:
            # scatter permutations
            size = np.arange(1, self.stat_null.shape[1] + 1)
            _size = np.repeat(np.atleast_2d(size), self.num_perm, axis=0)
            xy = np.vstack((_size.flatten(), self.stat_null.flatten())).T
            xy = xy[~np.isnan(xy[:, 1]), :]

            fig, ax = plt.subplots(1, 1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.scatter(xy[:, 0], xy[:, 1], color='g', alpha=.15,
                        label='observed')

            plot_thresh(p_list=[.05, .01, .001])
            plt.ylabel(f'max {self.stat} in permutation')
            plt.xlabel('size')
            plt.legend()

            save_fig(self.folder / f'size_v_{self.stat}_null.pdf')

        if performance and self.mask_target is not None:
            f_mask = self.folder / 'mask_target.nii.gz'
            self.mask_target.to_nii(f_mask)

            s = ''
            for label, mask in self.mode_est_mask_dict.items():
                s += get_performance_str(mask_estimate=mask,
                                         mask_target=self.mask_target,
                                         label=label) + '\n'

            print(s)
            with open(str(self.folder / 'performance.txt'), 'a+') as f:
                print(s, file=f)

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


def get_performance_str(mask_estimate, mask_target, label=None):
    mask_estimate = mask_estimate.astype(bool)
    mask_target = mask_target.astype(bool)
    dice_score = 1 - dice(mask_estimate.flatten(), mask_target.flatten())
    target = mask_target.sum()
    target_correct = (mask_target & mask_estimate).sum()
    sens = target_correct / target

    non_target = (~mask_target).sum()
    non_target_correct = (~mask_target & ~mask_estimate).sum()
    non_target_wrong = non_target - non_target_correct
    spec = non_target_correct / non_target

    s = f'---{label}---\n'
    s += f'dice: {dice_score:.3f}\n'
    s += f'sens: {sens:.3f} ({target_correct} of {target} vox detected correctly)\n'
    s += f'spec: {spec:.3f} ({non_target_wrong} of {non_target} vox detected incorrectly)\n'

    return s
