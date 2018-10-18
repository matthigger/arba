""" builds an effect boundry segmentation.

effects are assumed to be contained in volumes with some consistency in the
healthy population (they do not mix GM and WM, for example).  To model them, we
randomly sample a position in the brain and 'grow' (ie binary dilate) the
affected volume *within* some region.  This script builds a segmentation which
contains regions of similar feature (min cov)
"""

from collections import defaultdict

from pnl_data.set.cidar_post import folder, get_name
from pnl_segment.adaptive.part_graph_factory import part_graph_factory
from mh_pytools import file
import numpy as np

edge_per_step = 1e-2
gran_array = np.geomspace(5000, 50, 20).astype(int)
feat_tuple = 'fa', 'md'
folder_data = folder / 'dti_in_01193'
folder_out = folder_data / 'rba_regions'

# build f_img_tree
f_img_tree = defaultdict(dict)
for label in feat_tuple:
    for f in folder_data.glob(f'*{label}.nii.gz'):
        f_img_tree[get_name(f.stem)][label] = f

# build f_img_dict (grouped tree for part_graph)
f_img_dict = {'h': [[d[f] for f in feat_tuple] for d in f_img_tree.values()]}

# f_fa is an arbitrary fa image, used for ref space and masking
f_fa = next(iter(f_img_dict.values()))[0][0]

# segment
pg = part_graph_factory(obj='min_var', grp_to_min_var='h',
                        f_img_dict=f_img_dict, f_mask=f_fa, verbose=True)
for gran in gran_array:
    pg.reduce_to(gran, edge_per_step=edge_per_step, verbose=True)

    f_out = folder_out / f'rba_{gran}.p.gz'
    file.save(pg, file=f_out)

    f_out = folder_out / f'rba_{gran}.nii.gz'
    pg.to_nii(f_out, ref=f_fa)
