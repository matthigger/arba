from collections import defaultdict

import numpy as np

from mh_pytools import file
from arba.region import FeatStat
from pnl_data.set.hcp_100 import folder
from tqdm import tqdm

min_n = 10
folder = folder / f'arba_cv_MF_FA' / 'arba_permute'

permute = file.load(folder / 'permute.p.gz')
merge_record = permute.sg_hist.merge_record
file_tree = permute.file_tree

grp_sbj_dict = defaultdict(list)
for sbj in file_tree.sbj_list:
    grp_sbj_dict[sbj.gender].append(sbj)

n_cov_sbj_grp_dict = defaultdict(list)
reg_fs_list_dict = dict()
with file_tree.loaded():
    iter_sg = merge_record.get_iter_sg(file_tree, grp_sbj_dict)

    # initial
    sg, _, _ = next(iter_sg)

    for _, reg_new, reg_last in tqdm(iter_sg, total=len(sg)):
        if len(reg_new) < min_n:
            continue

        try:
            # compute new fs_list
            reg0, reg1 = reg_last
            fs_list = [fs0 + fs1 for fs0, fs1 in zip(reg_fs_list_dict[reg0],
                                                     reg_fs_list_dict[reg1])]

            # update dict
            reg_fs_list_dict[reg_new] = fs_list
            del reg_fs_list_dict[reg0]
            del reg_fs_list_dict[reg1]

        except KeyError:
            # we haven't seen reg_last before, compute from scratch
            fs_list = list()
            for sbj_idx, _ in enumerate(file_tree.sbj_list):
                mask = reg_new.pc_ijk.to_mask()
                x = file_tree.data[mask, sbj_idx, :].T
                fs_list.append(FeatStat.from_array(x))

            reg_fs_list_dict[reg_new] = fs_list

        n = len(reg_new)

        # store
        det_sbj = np.mean([fs.cov for fs in fs_list])
        det_grp = reg_new.cov_pooled[0, 0]
        n_cov_sbj_grp_dict[n].append((det_sbj, det_grp))


file.save(n_cov_sbj_grp_dict, folder / 'n_cov_sbj_grp_dict.p.gz')