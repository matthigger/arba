import pathlib
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import dice
from tqdm import tqdm

import arba

dim_sbj = 1
dim_img = 1

# subject params
mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 50

# imaging params
mu_img = np.zeros(dim_img)
sig_img = np.eye(dim_img)
shape = 6, 6, 6

# detection params
num_perm = 3

# regression params
r2 = .5
mask = np.zeros(shape)
mask[2:5, 2:5, 2:5] = True

# build effect
ref = arba.space.RefSpace(affine=np.eye(4))
mask = arba.space.Mask(mask, ref=ref)

# build output folder
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
folder.mkdir()
print(folder)
shutil.copy(__file__, folder / 'ex.py')

# sample sbj features
feat_sbj = np.random.multivariate_normal(mean=mu_sbj,
                                         cov=sig_sbj,
                                         size=num_sbj)

# build feat_img (shape0, shape1, shape2, num_sbj, dim_img)
feat_img = np.random.multivariate_normal(mean=mu_img,
                                         cov=sig_img,
                                         size=(*shape, num_sbj))

# build file_tree
file_tree = arba.data.SynthFileTree.from_array(data=feat_img,
                                               folder=folder / 'data')

feat_mapper = arba.regress.FeatMapperStatic(n=dim_sbj,
                                            sbj_list=file_tree.sbj_list,
                                            feat_sbj=feat_sbj)

# build regression, impose it
with file_tree.loaded():
    fs = file_tree.get_fs(mask=mask)
eff = arba.simulate.EffectRegress.from_r2(r2=r2,
                                          mask=mask,
                                          eps_img=fs.cov,
                                          cov_sbj=np.cov(feat_sbj.T, ddof=0),
                                          feat_mapper=feat_mapper)


#
def mse(reg, **kwargs):
    return reg.mse


def weighted_r2(reg, **kwargs):
    return reg.r2 * len(reg)


f_mask = folder / 'target_mask.nii.gz'
mask.to_nii(f_mask)


def run(feat_sbj, file_tree, fnc_target, save_folder=None, max_flag=True,
        cutoff_perc=95, fnc_tuple=None, **kwargs):
    # ensure that fnc_target in fnc_tuple
    if fnc_tuple is None:
        fnc_tuple = fnc_target,
    elif fnc_target not in fnc_tuple:
        fnc_tuple = *fnc_tuple, fnc_target

    if not max_flag:
        # in min mode, we need 'inverse' of cutoff_perc
        cutoff_perc = 100 - cutoff_perc

    with file_tree.loaded():
        sg_hist = run_single(feat_sbj, file_tree, fnc_tuple=fnc_tuple,
                             **kwargs)

        # identify cutoff for signifigance
        val_list = run_permute(fnc_target, max_flag=max_flag,
                               feat_sbj=feat_sbj, file_tree=file_tree,
                               fnc_tuple=fnc_tuple, **kwargs)
        cutoff = np.percentile(val_list, cutoff_perc)

        # get nodes in hierarchy which (greedily) maximize fnc_target
        merge_record = sg_hist.merge_record
        node_fnc_dict = merge_record.fnc_node_val_list[fnc_target]
        node_list = sg_hist._cut_greedy(node_fnc_dict, max_flag=max_flag)
        reg_list = list()
        for n in node_list:
            if max_flag and node_fnc_dict[n] < cutoff:
                # node is not significant
                continue
            elif not max_flag and node_fnc_dict[n] > cutoff:
                # node is not significant
                continue

            reg = merge_record.resolve_node(node=n,
                                            file_tree=file_tree,
                                            reg_cls=arba.region.RegionRegress)
            reg_list.append(reg)

        if save_folder:
            merge_record.to_nii(folder / 'seg_hier.nii.gz', n=10)

            # histogram of permutation testing
            plt.hist(val_list, bins=100)
            plt.xlabel(f'max {fnc_target.__name__} per permutations')
            plt.suptitle(f'{len(val_list)} permutations')
            arba.plot.save_fig(folder / 'permute_nr2.pdf')

            for idx, r in enumerate(reg_list):
                r.plot()
                f_out = save_folder / f'reg_{idx}_scatter.pdf'
                arba.plot.save_fig(f_out=f_out)

                f_out = save_folder / f'reg_{idx}_mask.nii.gz'
                r.pc_ijk.to_mask().to_nii(f_out=f_out)

    return sg_hist, reg_list, val_list


def run_single(feat_sbj, file_tree, fnc_tuple=None, permute_seed=None,
               **kwargs):
    if fnc_tuple is None:
        fnc_tuple = tuple()

    # set feat_sbj
    arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                           sbj_list=file_tree.sbj_list)

    if permute_seed:
        arba.region.RegionRegress.shuffle_feat_sbj(seed=permute_seed)

    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=file_tree,
                                             cls_reg=arba.region.RegionRegress,
                                             fnc_tuple=fnc_tuple)

    sg_hist.reduce_to(1, verbose=True)

    return sg_hist


def run_permute(fnc_target, feat_sbj, file_tree, n=5000, max_flag=True,
                **kwargs):
    val_list = list()
    for _n in tqdm(range(n), desc='permute'):
        sg_hist = run_single(feat_sbj, file_tree, permute_seed=_n + 1,
                             **kwargs)
        node_val_dict = sg_hist.merge_record.fnc_node_val_list[fnc_target]
        if max_flag:
            val = max(node_val_dict.values())
        else:
            val = min(node_val_dict.values())
        val_list.append(val)

    # ensure that RegionRegress has appropriate feat_sbj ordering
    arba.region.RegionRegress.set_feat_sbj(feat_sbj=feat_sbj,
                                           sbj_list=file_tree.sbj_list)

    return val_list


def compute_print_dice(reg_list, mask_target, folder):
    # build mask of detected area
    if reg_list:
        mask_detected = sum(r.pc_ijk.to_mask() for r in reg_list).astype(bool)
    else:
        # no regions are sig
        mask_detected = np.zeros(shape=mask_target.shape).astype(bool)

    # compute / record dice
    dice_score = 1 - dice(mask_detected.flatten(), mask_target.flatten())
    with open(str(folder / 'dice.txt'), 'w') as f:
        print(f'dice is {dice_score:.3f}', file=f)
        print(f'target vox: {mask_target.sum()}', file=f)
        print(f'detected vox: {mask_detected.sum()}', file=f)
        true_detect = (mask_target & mask_detected).sum()
        print(f'true detected vox: {true_detect}', file=f)
        false_detect = (~mask_target & mask_detected).sum()
        print(f'false detected vox: {false_detect}', file=f)

    return dice_score


fnc_tuple = mse, weighted_r2
with file_tree.loaded(effect_list=[eff]):
    sg_hist, reg_list, val_list = run(feat_sbj, file_tree,
                                      fnc_target=weighted_r2,
                                      save_folder=folder, max_flag=True,
                                      cutoff_perc=95, n=num_perm,
                                      fnc_tuple=fnc_tuple)

node_mask, d_max = sg_hist.merge_record.get_node_max_dice(mask)

sg_hist.merge_record.plot_size_v(weighted_r2, label='n * r2', mask=mask,
                                 log_y=True)
arba.plot.save_fig(folder / 'size_v_nr2.pdf')

sg_hist.merge_record.plot_size_v(mse, label='mse', mask=mask)
arba.plot.save_fig(folder / 'size_v_mse.pdf')

compute_print_dice(reg_list, mask_target=mask, folder=folder)
print(folder)
