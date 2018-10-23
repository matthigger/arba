import os
import pathlib
import random
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from mh_pytools import file
from pnl_segment.adaptive.part_graph_factory import part_graph_factory
from pnl_segment.point_cloud.ref_space import RefSpace
from .effect import EffectDm
from .mask import Mask


def get_folder(folder=None):
    if folder is None:
        folder = tempfile.TemporaryDirectory().name
    folder = pathlib.Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def run_multi(fnc):
    """ decorator, allows fnc to be called multiple times in parallel
    """

    def fnc_multi(sim, n=1, verbose=False, **kwargs):
        res_list = list()
        tqdm_dict = {'desc': fnc.__name__,
                     'disable': not verbose,
                     'total': n}
        for _ in tqdm(range(n), **tqdm_dict):
            res_list.append(fnc(sim, verbose=verbose, **kwargs))

        if n > 1:
            return res_list
        else:
            return res_list[0]

    return fnc_multi


class Simulator:
    """ generates an effect, and applies it to random healthy, outputs nii

    a healthy img is generated by sampling the given healthy population.
    an effect img is generated by adding some constant vector to same mask of a
    healthy image (see effect.py)

    Attributes:
        f_img_health (dict): first keys are sbj, second keys are names of feat,
                           leafs are names of files which contain img of
                           sbj's feat (all assumed in registered space)
    """

    def __init__(self, f_img_health):
        self.f_img_health = f_img_health

        # check ref spaces, store a copy
        ref_list = [RefSpace.from_nii(f) for f in self.iter_img()]
        for r in ref_list[1:]:
            if ref_list[0] != r:
                raise AttributeError('inconsistent reference space')
        self.ref_space = ref_list[0]

    def iter_img(self, feat_iter=None, sbj_iter=None):
        """ iterates over all healthy img"""
        if feat_iter is None:
            sbj = next(iter(self.f_img_health.keys()))
            feat_iter = sorted(self.f_img_health[sbj])

        if sbj_iter is None:
            sbj_iter = self.f_img_health.keys()

        for feat in feat_iter:
            for sbj in sbj_iter:
                yield self.f_img_health[sbj][feat]

    def split_sbj(self, eff_ratio=.5, seed=datetime.now(), **kwargs):
        """ splits sbj into effect and healthy groups

        Args:
            eff_ratio (float): in (0, 1), determines how many sbj have effect
            seed: seed given to randomize

        Returns:
            sbj_effect: sbj chosen to have effect
            sbj_health: remaining sbj
        """
        if 'sbj_effect' in kwargs.keys() or 'sbj_health' in kwargs.keys():
            # if split is already passed, pass it back (see run_effect())
            sbj_effect = kwargs['sbj_effect']
            sbj_health = kwargs['sbj_health']
            return sbj_effect, sbj_health

        # compute n_effect
        n_sbj = len(self.f_img_health.keys())
        n_effect = np.ceil(eff_ratio * n_sbj).astype(int)

        random.seed(seed)

        # split sbj into health and effect groups
        sbj_all = set(self.f_img_health.keys())
        sbj_effect = random.sample(sbj_all, k=n_effect)
        sbj_health = sbj_all - set(sbj_effect)

        return sbj_effect, sbj_health

    def sample_eff(self, effect, sbj_effect, sbj_health, folder=None,
                   sym_link_health=True, save=True,
                   label='{sbj}_{feat}_{eff_label}.nii.gz', **kwargs):
        """ adds effect to eff_ratio (without replacement) of healthy img

        Args:
            effect (Effect): applies effect to img set
            sbj_effect: sbj chosen to have effect
            sbj_health: remaining sbj
            folder (str or Path): path to stored effect files
            sym_link_health (bool): if True, healthy img are sym linked (makes
                                    for clean folder structure)
            save (bool): toggles saving mask / effect to folder
            label (str): name of effect files to produce

        Returns:
            f_img_dict (dict): keys are grp labels, values are iter, each
                               element of the iter is a list of img from a sbj
                               this is confusing, here's an example:
                               {'eff': [['sbj1_FA.nii.gz', 'sbj1_MD.nii.gz'], \
                                        ['sbj2_FA.nii.gz', 'sbj2_MD.nii.gz']],
                                'healthy': ... }
        """
        # get folder
        folder = get_folder(folder)

        def get_f_nii_out(sbj, eff_label):
            # find locations for output files
            f_nii_dict_out = dict()
            for feat in effect.feat_label:
                _label = label.format(sbj=sbj, feat=feat, eff_label=eff_label)
                f_nii_dict_out[feat] = folder / _label

            # sorted in same order as effect.feat_label
            img_list = [f_nii_dict_out[feat] for feat in effect.feat_label]

            return f_nii_dict_out, img_list

        # init f_img_dict with healthies
        f_img_dict = defaultdict(list)
        for sbj in sbj_health:
            img_list = [self.f_img_health[sbj][f] for f in effect.feat_label]
            if sym_link_health:
                # build sym links, replace img_list with these sym_links
                _, img_list_sym = get_f_nii_out(sbj, eff_label='healthy')
                for f, f_sym in zip(img_list, img_list_sym):
                    os.symlink(f, f_sym)
                img_list = img_list_sym
            f_img_dict['healthy'].append(img_list)

        # add sbj_effect to f_img_dict
        for sbj in sbj_effect:
            f_nii_dict_out, img_list = get_f_nii_out(sbj, eff_label='effect')

            # apply effect
            effect.apply_from_to_nii(f_nii_dict=self.f_img_health[sbj],
                                     f_nii_dict_out=f_nii_dict_out)

            f_img_dict[effect].append(img_list)

        if save and hasattr(effect, 'mask'):
            effect.mask.to_nii(f_out=folder / 'effect_mask.nii.gz',
                               f_ref=next(self.iter_img()))
            file.save(effect, file=folder / 'effect.p.gz')

        return f_img_dict, folder

    def _run(self, f_img_dict, folder, obj, f_mask=None,
             verbose=False, mask_to_nii=True, save=False, **kwargs):
        """ runs experiment
        """
        # get location of mask output file (if needed)
        if mask_to_nii:
            f_mask_out = folder / 'mask_active.nii.gz'
        else:
            # dummy value, mask.to_nii() will get temp location
            f_mask_out = None

        # build (or symlink) mask file
        if f_mask is None:
            # build mask as intersection of all img, save to file
            mask = Mask.build_intersection_from_nii(self.iter_img(),
                                                    thresh=.95)
            f_mask = mask.to_nii(f_out=f_mask_out)
        elif mask_to_nii:
            # mask file already exists, build symlink
            os.symlink(f_mask, f_mask_out)

        # build part graph
        pg_hist = part_graph_factory(obj=obj, f_img_dict=f_img_dict,
                                     history=True, f_mask=f_mask,
                                     verbose=verbose, **kwargs)

        if save:
            def get_obj(reg):
                return -reg.obj

            file.save(pg_hist, folder / 'pg_init.p.gz')
            pg_hist.to_nii(f_out=folder / f'_{obj}_vba.nii.gz',
                           ref=f_mask,
                           fnc=get_obj)

        # reduce
        pg_hist.reduce_to(1, verbose=verbose)

        # build segmentation
        def spanning_fnc(reg):
            return reg.obj

        pg_span = pg_hist.get_min_spanning_region(spanning_fnc)

        if save:
            file.save(pg_span, file=folder / 'pg_span.p.gz')

            file.save(pg_hist, file=folder / 'pg_hist.p.gz')

            pg_span.to_nii(f_out=folder / f'_{obj}_arba.nii.gz',
                           ref=f_mask,
                           fnc=get_obj)

        return pg_span, pg_hist

    @run_multi
    def run_healthy(self, folder=None, **kwargs):
        folder = get_folder(folder)

        # build feat_label
        sbj = next(iter(self.f_img_health.keys()))
        feat_label = sorted(self.f_img_health[sbj].keys())

        # build dummy effect
        effect_dm = EffectDm(feat_label=feat_label)

        # apply 'effect' to some subset of images
        kwargs['sbj_effect'], kwargs['sbj_health'] = self.split_sbj(**kwargs)
        f_img_dict, _ = self.sample_eff(effect_dm, folder=folder, **kwargs)
        return self._run(f_img_dict, folder=folder, **kwargs), folder

    @run_multi
    def run_effect(self, effect=None, folder=None, **kwargs):
        if effect is None:
            raise AttributeError('effect required')

        folder = get_folder(folder)

        kwargs['sbj_effect'], kwargs['sbj_health'] = self.split_sbj(**kwargs)
        f_img_dict, _ = self.sample_eff(effect, folder=folder, **kwargs)
        return self._run(f_img_dict, folder=folder, **kwargs), folder


def get_f_img_dict(folder, feat_list=None):
    """ gets f_img_dict from all files in folder

    assumes files are named in format:
    
    {sbj}_{feat}_{eff_label}.nii.gz

    and that same feat are available per sbj
    """

    f_img = '{sbj}_{feat}_{label}.nii.gz'

    sbj_dict = defaultdict(set)
    feat_set = set()
    for f in folder.glob('*.nii.gz'):
        try:
            sbj, feat, label = str(f.stem).split('.')[0].split('_')
            sbj_dict[label].add(sbj)
            feat_set.add(feat)
        except ValueError:
            continue

    if feat_list is None:
        feat_list = sorted(feat_set)
    elif set(feat_list) != feat_set:
        raise AttributeError('feat_list must have same elements as feat_set')

    f_img_dict = defaultdict(list)
    for label, sbj_set in sbj_dict.items():
        for sbj in sbj_set:
            f_img_list = list()
            for feat in feat_list:
                f = folder / f_img.format(sbj=sbj, feat=feat, label=label)
                if not f.exists():
                    raise FileExistsError
                f_img_list.append(f)
            f_img_dict[label].append(f_img_list)

    return f_img_dict, feat_list
