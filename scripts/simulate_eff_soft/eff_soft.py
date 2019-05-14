from arba.data import FileTree
from pnl_data.set import hcp_100
from collections import defaultdict
from arba.space import Mask, sample_mask
from arba.simulate import Effect
feat_tuple = 'fa', 'md'
num_vox = 3000

eff_num_vox = 1000
eff_edge_n = 2
eff_t2 = 4

# build file tree
folder = hcp_100.folder / 'to_100307_low_res'
sbj_feat_file_tree = defaultdict(dict)
for feat in feat_tuple:
    for f in folder.glob(f'*_{feat.upper()}.nii.gz'):
        sbj = f.stem.split('_')[0]
        try:
            int(sbj)
        except ValueError:
            # not a sbj, each sbj is numeric
            continue
        sbj_feat_file_tree[sbj][feat] = f

# build mask, sample to num_vox
mask = Mask.from_nii(f_nii=folder / 'mean_FA.nii.gz')
mask = sample_mask(prior_array=mask,
                   num_vox=num_vox,
                   ref=mask.ref)
file_tree = FileTree(sbj_feat_file_tree=sbj_feat_file_tree, mask=mask)

# build effects
with file_tree.loaded():
    eff_mask = sample_mask(prior_array=mask,
                           num_vox=eff_num_vox,
                           ref=mask.ref)
    fs = file_tree.get_fs(mask=eff_mask)
    eff = Effect.from_fs_t2(fs, t2=eff_t2, mask=eff_mask, edge_n=eff_edge_n)



# test each method

# save
