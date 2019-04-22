from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from tqdm import tqdm

from arba.plot import save_fig, size_v_pval
from arba.region import RegionWardSbj
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

# load
folder = folder / 'arba_cv_MF_FA_test' / 'arba_permute'

f_sg_old = folder / 'sg_old_t2.p.gz'
permute = file.load(folder / 'permute.p.gz')
file_tree = permute.file_tree

sg = file.load(f_sg_old)
min_pval = np.inf
for reg in tqdm(sg.nodes):
    if reg.pval < min_pval:
        reg_min = reg
        min_pval = reg.pval
reg_list = [reg_min] + \
           list(nx.descendants(sg, reg_min)) + \
           list(nx.ancestors(sg, reg_min))

grp_sbj_dict = defaultdict(list)
for sbj in file_tree.sbj_list:
    grp_sbj_dict[sbj.gender].append(sbj)

reg_list_new = list()
node_map = dict()
split = np.array([sbj.gender == 'M' for sbj in file_tree.sbj_list])
with file_tree.loaded():
    for reg in tqdm(reg_list, desc='converting'):
        reg_new = RegionWardSbj.from_data(file_tree=file_tree,
                                          pc_ijk=reg.pc_ijk,
                                          fs_dict=reg.fs_dict,
                                          grp_sbj_dict=grp_sbj_dict)
        node_map[reg] = reg_new
        reg_list_new.append(reg_new)

sg_new = nx.relabel_nodes(sg, node_map, copy=True)

file.save((sg, sg_new, reg_list, reg_list_new, node_map), folder / 'temp.p.gz')

# sg, sg_new, reg_list, reg_list_new, node_map = file.load(folder / 'temp.p.gz')


# pval
fig, ax = plt.subplots(1, 1)
size_v_pval(sg_new, min_reg_size=10, reg_list=reg_list_new, ax=ax,
            node_color='r', alpha=.2)
size_v_pval(sg, min_reg_size=10, reg_list=reg_list, ax=ax, node_color='b',
            alpha=.2)
print(save_fig(f_out=folder / f'history_min_pval_both.pdf'))

fig, ax = plt.subplots(1, 1)
size_v_pval(sg_new, min_reg_size=10, reg_list=reg_list_new, ax=ax,
            node_color='r')
print(save_fig(f_out=folder / f'history_min_pval_new.pdf'))

fig, ax = plt.subplots(1, 1)
size_v_pval(sg, min_reg_size=10, reg_list=reg_list, ax=ax, node_color='b')
print(save_fig(f_out=folder / f'history_min_pval_old.pdf'))

# variance
reg = min(reg_list_new, key=len)
reg_list_new = sorted(nx.ancestors(sg_new, reg), key=len)
n = [len(r) for r in reg_list_new]

m = np.mean([fs.n for fs in reg.fs_dict.values()])
cov_pooled = [r.cov_pooled[0][0] for r in reg_list_new]
sig_space = [np.mean(list(r.sig_space_dict.values())) for r in reg_list_new]
sig_sbj = [np.mean(list(r.sig_sbj_dict.values())) for r in reg_list_new]
sig_sum = [sum(x) for x in zip(sig_space, sig_sbj)]
sig_space_over_nm = [v / (_n * m) for v, _n in zip(sig_space, n)]
cov_pooled_over_nm = [v / (_n * m) for v, _n in zip(cov_pooled, n)]
sig_sbj_over_m = [v / m for v in sig_sbj]

fig, ax = plt.subplots(1, 1)
sns.set(font_scale=1.2)
plt.plot(n, cov_pooled, label='$\Sigma$', linewidth=2, color='r')
plt.plot(n, sig_space, label='$\Sigma_{space}$', linewidth=2, color='b')
plt.plot(n, sig_sbj, label='$\Sigma_{sbj}$', linewidth=2, color='g')

plt.plot(n, cov_pooled_over_nm, label='$\Sigma / nm$', linewidth=2, color='r',
         linestyle='--')
plt.plot(n, sig_space_over_nm, label='$\Sigma_{space} / nm$', linewidth=2,
         color='b', linestyle='--')
plt.plot(n, sig_sbj_over_m, label='$\Sigma_{sbj} / n$', linewidth=2, color='g',
         linestyle='--')

plt.legend()
plt.xlabel('region size (voxels)')
plt.ylabel('variance of feature (FA)')
ax.set_yscale('log')
ax.set_xscale('log')
print(save_fig(f_out=folder / f'var_vs_size.pdf'))
