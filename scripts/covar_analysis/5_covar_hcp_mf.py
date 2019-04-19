from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from arba.region import RegionWardSbj
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

# load
folder = folder / 'arba_cv_MF_FA_test' / 'arba_permute'

f_sg_old = folder / 'sg_old_t2.p.gz'
f_sg_new = folder / 'sg_new_t2.p.gz'

permute = file.load(folder / 'permute.p.gz')
merge_record = permute.sg_hist.merge_record
file_tree = permute.file_tree

grp_sbj_dict = defaultdict(list)
for sbj in file_tree.sbj_list:
    grp_sbj_dict[sbj.gender].append(sbj)

# # resolve (old t2)
# with file_tree.loaded():
#     sg = merge_record.resolve_hist(file_tree, grp_sbj_dict)
#
# sg_old_t2 = sg
# file.save(sg, f_sg_old)

# map to new t2 style
sg_old_t2, _ = file.load(f_sg_old)
node_map = dict()
split = np.array([sbj.gender == 'M' for sbj in file_tree.sbj_list])
with file_tree.loaded():
    for reg in tqdm(sg_old_t2.nodes, desc='converting', total=len(sg_old_t2)):
        reg_new = RegionWardSbj.from_data(file_tree=file_tree,
                                          pc_ijk=reg.pc_ijk,
                                          fs_dict=reg.fs_dict,
                                          grp_sbj_dict=grp_sbj_dict)
        node_map[reg] = reg_new

sg_new_t2 = nx.relabel_nodes(sg_old_t2, node_map, copy=True)

print('saving')
file.save(sg_new_t2, f_sg_new)
