import pathlib
import random
import shutil
from collections import defaultdict

import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba import permute
from mh_pytools import file
from mh_pytools import parallel
from .effect import Effect
from ..space import sample_mask, sample_mask_min_var, PointCloud

method_permute_dict = {'arba': permute.PermuteARBA,
                       'tfce': permute.PermuteTFCE,
                       'ptfce': permute.PermutePTFCE}


class Simulator:
    """ manages sampling effects and detecting via methods
    """

    f_performance = 'performance_stats.p.gz'

    def __init__(self, folder, file_tree, p_effect=.5, effect_shape='min_var',
                 verbose=True, par_flag=True, num_perm=5000, alpha=.05,
                 method_list=None, active_rad=None, print_image=False):

        self.folder = pathlib.Path(folder)
        if self.folder.exists():
            shutil.rmtree(self.folder)
        folder.mkdir(parents=True)

        # split into two file_trees
        self.file_tree = file_tree
        self.verbose = verbose
        self.effect_shape = effect_shape
        self.effect_list = list()
        self.par_flag = par_flag
        self.active_rad = active_rad

        # split defines who is in the effect grp, fixed for all simulations
        n_effect = int(self.file_tree.num_sbj * p_effect)
        ones_idx = np.random.choice(range(self.file_tree.num_sbj),
                                    size=n_effect,
                                    replace=False)
        self.split = tuple(idx in ones_idx
                           for idx in range(self.file_tree.num_sbj))

        # debug: to rm
        n_eff = int(self.file_tree.num_sbj / 2)
        n_no_eff = self.file_tree.num_sbj - n_eff
        self.split = tuple(([False] * n_eff) + ([True] * n_no_eff))

        # comparison parameters
        if method_list is None:
            method_list = ['arba']
        self.method_list = method_list
        self.num_perm = num_perm
        self.alpha = alpha

        # output params
        self.print_image = print_image

    def build_effect_list(self, radius=None, num_vox=None, seg_array=None,
                          seed=1):

        # reset seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        with self.file_tree.loaded():

            # build input list
            arg_list = list()
            if self.effect_shape == 'min_var':
                fnc_sample_mask = sample_mask_min_var
                # minimum variance effect regions
                for n in num_vox:
                    d = {'file_tree': self.file_tree,
                         'num_vox': n}
                    arg_list.append(d)

            elif self.effect_shape == 'cube':
                # cubic effect regions
                fnc_sample_mask = sample_mask
                if (radius is None) == (num_vox is None):
                    raise AttributeError('either radius xor num_vox required')
                elif radius is not None:
                    for rad in radius:
                        d = {'prior_array': self.file_tree.mask,
                             'ref': self.file_tree.ref,
                             'radius': rad,
                             'seg_array': seg_array}
                        arg_list.append(d)
                else:
                    for n in num_vox:
                        d = {'prior_array': self.file_tree.mask,
                             'ref': self.file_tree.ref,
                             'num_vox': n,
                             'seg_array': seg_array}
                        arg_list.append(d)

            # sample mask
            tqdm_dict = {'desc': 'sample effect mask',
                         'disable': not self.verbose}
            if self.par_flag:
                mask_list = parallel.run_par_fnc(fnc_sample_mask, arg_list,
                                                 desc=tqdm_dict['desc'])
            else:
                mask_list = list()
                for d in tqdm(arg_list, **tqdm_dict):
                    mask_list.append(fnc_sample_mask(**d))

            # build effects (such that their locations are constant across t2)
            self.effect_list = list()
            for mask in mask_list:
                pc = PointCloud.from_mask(mask)
                fs = sum(self.file_tree.get_fs(ijk) for ijk in pc)

                # t2 below is placeholder, it is reset with each run
                e = Effect.from_fs_t2(fs=fs, mask=mask, t2=1)
                self.effect_list.append(e)

    def run_effect_permute(self, effect, t2=None, folder=None, **kwargs):
        # get mask of active area
        if self.active_rad is not None:
            mask_eff_dilated = effect.mask.dilate(self.active_rad)
            self.file_tree.mask = np.logical_and(mask_eff_dilated,
                                                 self.file_tree.mask)

        # set scale of effect
        if t2 is not None:
            effect.t2 = t2

        # modify file_tree to specific effect
        self.file_tree.split_effect = (self.split, effect)

        # prep folder / save effect
        if folder is not None:
            file.save(effect, folder / 'effect.p.gz')
            f_out = folder / 'effect.nii.gz'
            effect.mask.to_nii(f_out=f_out)

            f_out = folder / 'mask_active.nii.gz'
            self.file_tree.mask.to_nii(f_out=f_out)

        _folder = None
        for method in self.method_list:
            if folder is not None:
                _folder = folder / method
                _folder.mkdir(exist_ok=True, parents=True)

            permute = method_permute_dict[method](self.file_tree)
            permute.run(self.split, n=self.num_perm, folder=_folder, **kwargs)

    def run(self, t2_list, **kwargs):

        # build arg_list
        arg_list = list()
        z_width_t2 = np.ceil(np.log10(len(t2_list))).astype(int)
        z_width_eff = np.ceil(np.log10(len(self.effect_list))).astype(int)
        for t2_idx, t2 in enumerate(sorted(t2_list)):
            for eff_idx, effect in enumerate(self.effect_list):
                s_t2 = str(t2_idx).zfill(z_width_t2)
                s_eff = str(eff_idx).zfill(z_width_eff)
                folder = self.folder / f't2_{s_t2}_{t2:.1E}_effect{s_eff}'

                d = {'effect': effect,
                     't2': t2,
                     'verbose': self.verbose and (not self.par_flag),
                     'folder': folder,
                     'par_flag': False}
                d.update(kwargs)
                arg_list.append(d)

        # run
        desc = f'simulating effects'
        if self.par_flag:
            parallel.run_par_fnc('run_effect_permute', arg_list=arg_list,
                                 obj=self, desc=desc)
        if not self.par_flag:
            for d in tqdm(arg_list, desc=desc, disable=not self.verbose):
                self.run_effect_permute(**d)

    def get_performance(self, alpha=.05, print_perf=True):
        t2size_method_ss_tree = defaultdict(lambda: defaultdict(list))

        for _folder in self.folder.glob('*t2*'):
            effect = file.load(_folder / 'effect.p.gz')
            mask = nib.load(str(_folder / 'mask_active.nii.gz')).get_data()

            for folder_method in _folder.glob('*'):
                if not folder_method.is_dir():
                    continue
                method = folder_method.stem

                # estimate is all pval which are <= alpha
                f_pval = folder_method / 'pval.nii.gz'
                pval = nib.load(str(f_pval)).get_data()
                estimate = np.logical_and(pval <= alpha, mask)

                sens, spec = effect.get_sens_spec(estimate, mask)

                t2_size = np.round(effect.t2, 3), int(effect.mask.sum())
                t2size_method_ss_tree[t2_size][method].append((sens, spec))

                if print_perf:
                    f_out = folder_method / 'detect_stats.txt'
                    with open(str(f_out), 'w') as f:
                        print(f'sensitivity: {sens:.3f}', file=f)
                        print(f'specificity: {spec:.3f}', file=f)

        # save performance stats
        f_performance = self.folder / self.f_performance
        file.save(dict(t2size_method_ss_tree), file=f_performance)

        return f_performance


def get_f(folder, f):
    """ there may be multiple copies of f within the folder (different
    subfolders: arba_cv and arba_permute).  any of them are valid, we just grab
    the first one ...
    todo: md5, check same
    """
    f = next(folder.glob(f'**/{f}'))
    return file.load(str(f))
