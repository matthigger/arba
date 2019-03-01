import copy
import pathlib
import random
import shutil
from collections import defaultdict

import nibabel as nib
import numpy as np
from tqdm import tqdm

from mh_pytools import file
from mh_pytools import parallel
from .effect import Effect
from .tfce import compute_tfce
from ..seg_graph import run_arba_cv, run_arba_permute
from ..space import PointCloud, sample_mask, sample_mask_min_var


class Simulator:
    """ manages sampling effects and detecting via methods

    todo: a bit monolithic ... should be refactored

    Attributes:
        folder (Path): location of output
        file_tree (FileTree): full file tree of all healthy sbj

        p_effect (float): percentage of sbj which have effect applied
        effect_list (list): list of effects to apply
    """

    grp_effect = 'EFFECT'
    grp_null = 'CONTROL'
    s_mask_sig = 'mask_sig_{method}.nii.gz'
    f_performance = 'performance_stats.p.gz'

    def __init__(self, folder, file_tree, p_effect=.5, effect_shape='min_var',
                 verbose=True, par_flag=True, f_rba=None, tfce_num_perm=5000,
                 alpha=.05):
        self.folder = pathlib.Path(folder)
        folder.mkdir(parents=True)

        # split into two file_trees
        self.file_tree = file_tree
        ft_eff, ft_null = file_tree.split(p=p_effect)
        self.ft_dict = {self.grp_effect: ft_eff,
                        self.grp_null: ft_null}
        self.verbose = verbose
        self.effect_shape = effect_shape
        self.effect_list = list()
        self.par_flag = par_flag

        # comparison parameters
        self.f_rba = f_rba
        self.tfce_num_perm = tfce_num_perm
        self.alpha = alpha

    def build_effect_list(self, radius=None, num_vox=None, seg_array=None,
                          seed=1):

        self.file_tree.load(verbose=self.verbose)

        # reset seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # build input list
        arg_list = list()
        if self.effect_shape == 'min_var':
            fnc_sample_mask = sample_mask_min_var
            # minimum variance effect regions
            for n in num_vox:
                d = {'ijk_fs_dict': self.file_tree.ijk_fs_dict,
                     'ref': self.file_tree.ref,
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
                mask_list.append(sample_mask(**d))

        # build effects (such that their locations are constant across t2)
        self.effect_list = list()
        for mask in mask_list:
            # compute feat_stat in mask
            pc = PointCloud.from_mask(mask)
            fs = sum(self.file_tree.ijk_fs_dict[ijk] for ijk in pc)

            # t2 below is placeholder, it is reset with each run
            e = Effect.from_fs_t2(fs=fs, mask=mask, t2=1)
            self.effect_list.append(e)

        # delete voxel wise stats
        self.file_tree.unload()

    def run_effect_prep(self, effect, t2=None, active_rad=None, **kwargs):
        # get mask of active area
        mask_active = self.file_tree.mask
        if active_rad is not None:
            # only work in a dilated region around the effect
            mask_eff_dilated = effect.mask.dilate(active_rad)
            mask_active = np.logical_and(mask_eff_dilated, mask_active)

        # set scale of effect
        if t2 is not None:
            effect.t2 = t2

        # build effect dict
        grp_effect_dict = {self.grp_effect: effect}

        return mask_active, grp_effect_dict

    def run_effect_permute(self, effect, folder, **kwargs):
        mask, grp_effect_dict = self.run_effect_prep(effect, **kwargs)

        run_arba_permute(mask=mask,
                         grp_effect_dict=grp_effect_dict,
                         folder=folder / 'arba_permute',
                         ft_dict=self.ft_dict,
                         alpha=self.alpha,
                         **kwargs)

    def run_effect_cv(self, effect, folder, **kwargs):
        mask, grp_effect_dict = self.run_effect_prep(effect, **kwargs)

        run_arba_cv(mask=mask,
                    grp_effect_dict=grp_effect_dict,
                    folder=folder / 'arba_cv',
                    ft_dict=self.ft_dict,
                    alpha=self.alpha,
                    **kwargs)

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
                     'folder': folder}
                d.update(kwargs)
                arg_list.append(d)

        # run
        desc = f'simulating effects'
        if self.par_flag:
            parallel.run_par_fnc('run_effect_cv', arg_list=arg_list,
                                 obj=self, desc=desc)
        if not self.par_flag:
            for d in tqdm(arg_list, desc=desc, disable=not self.verbose):
                self.run_effect_cv(**d)

    def run_tfce(self, folder):
        # load saved file tree
        ft_dict = get_f(folder, 'ft_dict_.p.gz')
        ft0, ft1 = ft_dict[Simulator.grp_null], \
                   ft_dict[Simulator.grp_effect]

        # compute tfce
        folder_tfce = folder / 'tfce'
        folder_tfce.mkdir(exist_ok=True)
        f_sig = folder_tfce / self.s_mask_sig.format(method='tfce')
        _, f_sig_list = compute_tfce((ft0, ft1), alpha=self.alpha,
                                     folder=folder_tfce,
                                     f_data=folder_tfce / 't2.nii.gz',
                                     num_perm=self.tfce_num_perm)
        shutil.copy(f_sig_list[1], str(f_sig))

    def run_vba_rba(self, folder):
        # load saved file tree
        ft_dict = get_f(folder, 'ft_dict_.p.gz')
        ft0, ft1 = ft_dict[Simulator.grp_null], \
                   ft_dict[Simulator.grp_effect]
        ft0.load(load_ijk_fs=True, load_data=False)
        ft1.load(load_ijk_fs=True, load_data=False)

        sg_hist_seg = get_f(folder, 'sg_hist_seg.p.gz')
        sg_vba_test, _, _ = next(iter(sg_hist_seg))
        sg_vba = sg_vba_test.from_file_tree_dict(ft_dict)
        sg_dict = {'vba': sg_vba}

        if self.f_rba is not None:
            sg_dict['rba'] = copy.deepcopy(sg_vba)
            sg_dict['rba'].combine_by_reg(self.f_rba)

        # output masks of detected volumes
        for method, sg in sg_dict.items():
            folder_method = folder / method
            folder_method.mkdir(exist_ok=True)
            sg_sig = sg.get_sig(alpha=self.alpha)
            f_out = folder_method / self.s_mask_sig.format(method=method)
            sg_sig.to_nii(f_out=f_out,
                          ref=ft0.ref,
                          fnc=lambda r: np.uint8(1),
                          background=np.uint8(0))

    def get_performance(self, folder):
        # load effect
        effect = get_f(folder, f'effect_{self.grp_effect}.p.gz')

        # compute performance
        method_ss_dict = dict()
        for folder_method in folder.glob('*'):
            if not folder_method.is_dir():
                continue

            method = folder_method.name
            f_estimate = folder_method / self.s_mask_sig.format(method=method)
            estimate = nib.load(str(f_estimate)).get_data()
            mask = self.file_tree.mask
            method_ss_dict[method] = effect.get_sens_spec(estimate=estimate,
                                                          mask=mask)

        f_out = folder / 'detect_stats.txt'
        with open(str(f_out), 'w') as f:
            for method, (sens, spec) in sorted(method_ss_dict.items()):
                print(f'{method}: sens {sens:.2f} spec {spec:.2f}', file=f)

        t2_round = round(effect.t2, 4)
        effect_size = int(effect.mask.sum())
        return (t2_round, effect_size), method_ss_dict

    def _run_effect_comparison(self, folder):
        self.run_tfce(folder)
        self.run_vba_rba(folder)
        return self.get_performance(folder)

    def run_effect_comparison(self):

        # find relevant folders, build inputs to run()
        arg_list = list()
        for folder in sorted(self.folder.glob('*t2*')):
            if not folder.is_dir():
                continue
            arg_list.append({'folder': folder})

        desc = 'compute rba, vba, tfce and performance per effect'
        fnc = lambda: defaultdict(list)
        t2size_method_ss_tree = defaultdict(fnc)
        if self.par_flag:
            res = parallel.run_par_fnc('_run_effect_comparison', arg_list,
                                       desc=desc, obj=self)
            for maha_size, method_ss_dict in res:
                for method, ss in method_ss_dict.items():
                    t2size_method_ss_tree[maha_size][method].append(ss)

        else:
            for d in tqdm(arg_list, desc=desc):
                maha_size, method_ss_dict = self._run_effect_comparison(**d)
                for method, ss in method_ss_dict.items():
                    t2size_method_ss_tree[maha_size][method].append(ss)
        # save
        f_out = self.folder / self.f_performance
        file.save(dict(t2size_method_ss_tree), f_out)

        return t2size_method_ss_tree


def get_f(folder, f):
    """ there may be multiple copies of f within the folder (different
    subfolders: arba_cv and arba_permute).  any of them are valid, we just grab
    the first one ...
    todo: md5, check same
    """
    f = next(folder.glob(f'**/{f}'))
    return file.load(str(f))
