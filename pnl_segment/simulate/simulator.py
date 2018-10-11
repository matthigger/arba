import os
import pathlib
import tempfile
import uuid
from collections import defaultdict

import numpy as np

from .effect import EffectDm


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

    def __init__(self, f_img_health, split_ratio=.5, mask_dilate=5):
        self.f_img_tree = f_img_health
        self.split_ratio = split_ratio
        self.mask_dilate = mask_dilate

    def sample_eff(self, effect, folder=None, sym_link_health=True,
                   label='{sbj}_{feat}_{eff_label}.nii.gz'):
        """ adds effect to split_ratio (without replacement) of healthy img

        Args:
            effect (Effect): applies effect to img set
            folder (str or Path): path to stored effect files
            sym_link_health (bool): if True, healthy img are sym linked (makes
                                    for clean folder structure)
            label (str): name of effect files to produce

        Returns:
            f_img_dict (dict): keys are grp labels, values are iter, each
                               element of the iter is a list of img from a sbj
                               this is confusing, here's an example:
                               {'eff': [['sbj1_FA.nii.gz', 'sbj1_MD.nii.gz'], \
                                        ['sbj2_FA.nii.gz', 'sbj2_MD.nii.gz']],
                                'healthy': ... }
        """

        def get_f_nii_out(sbj, eff_label):
            # find locations for output files
            f_nii_dict_out = dict()
            for feat in effect.feat_label:
                _label = label.format(sbj=sbj, feat=feat, eff_label=eff_label)
                f_nii_dict_out[feat] = folder / _label

            # sorted in same order as effect.feat_label
            img_list = [f_nii_dict_out[feat] for feat in effect.feat_label]

            return f_nii_dict_out, img_list

        # compute n_effect
        n_sbj = len(self.f_img_health.keys())
        n_effect = np.ceil(self.split_ratio * n_sbj)

        # split sbj into health and effect groups
        sbj_effect = np.random.choice(self.sbj_list, n_effect, replace=False)
        sbj_health = set(self.f_img_health.keys()) - set(sbj_effect)

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

        # if no folder passed, put new effect img in temp
        if folder is None:
            data_label = uuid.uuid4().hex[:6]
            folder = tempfile.TemporaryDirectory(prefix=data_label).name
            folder = pathlib.Path(folder)
            folder.mkdir()

        # add sbj_effect to f_img_dict
        for sbj in sbj_effect:
            f_nii_dict_out, img_list = get_f_nii_out(sbj, eff_label='effect')

            # apply effect
            self.effect.apply_from_to_nii(f_nii_dict=self.f_img_tree[sbj],
                                          f_nii_dict_out=f_nii_dict_out)

            f_img_dict[effect].append(img_list)

        return f_img_dict, data_label

    def _run(self, f_img_dict):
        """ runs experiment
        """
        # build part graph

        # reduce

        # build segmentation

        # write segmentation to nii

        return f_segment_nii

    def run_healthy(self):
        effect_dm = EffectDm()
        f_img_dict = self.sample_eff(effect_dm)
        return self._run(f_img_dict)

    def run_effect(self, effect):
        f_img_dict = self.sample_eff(effect)
        return self._run(f_img_dict)

    def compute_auc(self, f_img_dict, f_segment_nii, mask_sep):
        """ computes auc """
