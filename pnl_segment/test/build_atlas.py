""" builds an effect boundry segmentation.

effects are assumed to be contained in volumes with some consistency in the
healthy population (they do not mix GM and WM, for example).  To model them, we
randomly sample a position in the brain and 'grow' (ie binary dilate) the
affected volume *within* some region.  This script builds a segmentation which
contains regions of similar feature (min cov)
"""

from collections import defaultdict

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment.adaptive.part_graph_factory import min_var

edge_per_step = 100
num_reg_segment = 300
feat_tuple = 'fa', 'md'

# build f_img_tree
folder_data = folder / 'dti_in_01193'
f_img_tree = defaultdict(dict)
for label in feat_tuple:
    for f in folder_data.glob(f'*{label}.nii.gz'):
        f_img_tree[get_name(f.stem)][label] = f

# build f_img_dict (grouped tree for part_graph)
f_img_dict = {'h': [[d[f] for f in feat_tuple] for d in f_img_tree.values()]}

# f_fa is an arbitrary fa image, used for ref space and masking
f_fa = next(iter(f_img_dict.values()))[0][0]

# segment
pg = min_var(grp_to_min_var='h', f_img_dict=f_img_dict, f_mask=f_fa)
pg.reduce_to(num_reg_segment)

# save
f_out = folder_data / f'atlas_{num_reg_segment}.nii.gz'
pg.to_nii(f_out=f_out, ref=f_fa, edge_per_step=edge_per_step)
