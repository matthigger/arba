import matplotlib.pyplot as plt
import seaborn as sns
from mh_pytools import file
from pnl_data.set.hcp_100 import folder
from arba.plot import save_fig


folder = folder / f'arba_cv_MF_FA' / 'arba_permute'
n_cov_sbj_grp_dict = file.load(folder / 'n_cov_sbj_grp_dict.p.gz')
f_out = folder / 'cov_ratio_vs_size.pdf'

sns.set(font_scale=1.2)

n_rat_list = list()
for n, sbj_grp_list in n_cov_sbj_grp_dict.items():
    for sbj, grp in sbj_grp_list:
        n_rat_list.append((n, sbj / grp))

n_list = [x[0] for x in n_rat_list]
rat_list = [x[1] for x in n_rat_list]
plt.scatter(n_list, rat_list, alpha=.05)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel('num voxels in region')
plt.ylabel('sbj_pooled_cov / grp_pooled_cov')
print(save_fig(f_out=f_out))