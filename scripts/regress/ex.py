import pathlib
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import dice
from tqdm import tqdm

import arba

dim_sbj = 2
dim_img = 3

# subject params
mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 50

# imaging params
mu_img = np.zeros(dim_img)
sig_img = np.eye(dim_img)
shape = 6, 6, 6

# detection params
cutoff_perc = 95
num_perm = 10

# regression params
r2 = .95
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
                                          cov_sbj=np.cov(feat_sbj.T),
                                          feat_mapper=feat_mapper)


#
def mse(reg, **kwargs):
    return reg.mse


def weighted_r2(reg, **kwargs):
    return reg.r2 * len(reg)


f_mask = folder / 'target_mask.nii.gz'
mask.to_nii(f_mask)


def run(feat_sbj, file_tree, fnc_tuple=(mse, weighted_r2), permute_seed=None):
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


def permute_run(target_fnc, n=5000, max_flag=True, **kwargs):
    val_list = list()
    for _n in tqdm(range(n), desc='permute'):
        sg_hist = run(permute_seed=_n + 1, **kwargs)
        node_val_dict = sg_hist.merge_record.fnc_node_val_list[target_fnc]
        if max_flag:
            val = max(node_val_dict.values())
        else:
            val = min(node_val_dict.values())
        val_list.append(val)

    return val_list


with file_tree.loaded(effect_list=[eff]):
    val_list = permute_run(n=num_perm, target_fnc=weighted_r2, max_flag=True,
                           feat_sbj=feat_sbj, file_tree=file_tree)

    sg_hist = run(feat_sbj=feat_sbj, file_tree=file_tree)

    merge_record = sg_hist.merge_record
    merge_record.to_nii(folder / 'seg_hier.nii.gz', n=10)

    node_nr2_dict = merge_record.fnc_node_val_list[weighted_r2]
    node_list = sg_hist._cut_greedy(node_nr2_dict, max_flag=True)
    reg_list = [merge_record.resolve_node(node=n,
                                          file_tree=file_tree,
                                          reg_cls=arba.region.RegionRegress)
                for n in node_list]

    node_mask, d_max = sg_hist.merge_record.get_node_max_dice(mask)
    reg_mask = sg_hist.merge_record.resolve_node(node=node_mask,
                                                 file_tree=file_tree,
                                                 reg_cls=arba.region.RegionRegress)

cutoff = np.percentile(val_list, cutoff_perc)
plt.hist(val_list, bins=100)
plt.xlabel('max n r^2 per permutations')
plt.suptitle(f'{len(val_list)} permutations')
arba.plot.save_fig(folder / 'permute_nr2.pdf')

reg_list = [r for r in reg_list if weighted_r2(r) > cutoff]

sg_hist.merge_record.plot_size_v(weighted_r2, label='n * r2', mask=mask,
                                 log_y=True)
arba.plot.save_fig(folder / 'size_v_nr2.pdf')

sg_hist.merge_record.plot_size_v(mse, label='mse', mask=mask)
arba.plot.save_fig(folder / 'size_v_mse.pdf')

for idx, r in enumerate(reg_list):
    r.plot()
    arba.plot.save_fig(f_out=folder / f'reg_{idx}_scatter.pdf')
    r.pc_ijk.to_mask().to_nii(f_out=folder / f'reg_{idx}_mask.nii.gz')

mask_detected = sum(r.pc_ijk.to_mask() for r in reg_list).astype(bool)
dice_score = 1 - dice(mask_detected.flatten(), mask.flatten())
with open(str(folder / 'dice.txt'), 'w') as f:
    print(f'dice is {dice_score:.3f}', file=f)
    print(f'target region size: {mask.sum()}', file=f)
    print(f'detected region size: {mask_detected.sum()}', file=f)
    print(f'true detected region size: {(mask & mask_detected).sum()}', file=f)
    print(f'false detected region size: {(~mask & mask_detected).sum()}',
          file=f)

print(folder)
