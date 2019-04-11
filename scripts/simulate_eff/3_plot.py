from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arba.plot import save_fig
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

sns.set(font_scale=1.2)

folder = folder / 'result' / 'FAMD_n70_size250_nperm1000_cube_full'
t2size_method_ss_list_tree = file.load(folder / 'performance_stats.p.gz')
delta_dict = defaultdict(list)
_, ax_sens = plt.subplots()
_, ax_spec = plt.subplots()
ax_dict = {'specificity': (ax_spec, 1),
           'sensitivity': (ax_sens, 0)}
color_dict = {'arba': 'r',
              'tfce': 'b'}
label_dict = {'arba': 'ARBA',
              'tfce': 'TFCE'}
mean_dict = defaultdict(list)
for (t2, size), method_ss_list in sorted(t2size_method_ss_list_tree.items()):
    for method, ss_list in method_ss_list.items():
        for stat, (ax, idx) in ax_dict.items():
            plt.sca(ax)
            stat_list = [x[idx] for x in ss_list]
            x = np.ones(len(stat_list)) * t2
            plt.scatter(x, stat_list, color=color_dict[method], alpha=.4)

            # compute mean
            mean_dict[method, stat].append(np.mean(stat_list))

t2 = sorted(x[0] for x in t2size_method_ss_list_tree.keys())
for stat, (ax, idx) in ax_dict.items():
    plt.sca(ax)
    ax.set_xscale('log')
    for (method, _stat), x in mean_dict.items():
        if _stat != stat:
            continue
        plt.plot(t2, x, label=label_dict[method], color=color_dict[method])

    f_out = folder / (stat[:4] + '_scatter.pdf')
    stat = stat[0].upper() + stat[1:]
    plt.xlabel('$T^2(v)$')
    plt.ylabel(stat)
    plt.legend()
    print(save_fig(f_out=f_out, size_inches=(4, 3)))

# plt.scatter(delta_dict['specificity'], delta_dict['sensitivity'])
# plt.xlabel('spec')
# plt.ylabel('sens')
#
# print(save_fig())
#
#
# sns.swarmplot(y=delta_dict['specificity'], color='r')
# plt.ylabel('ARBA Specificity Advantage')
# print(save_fig(size_inches=(3, 4)))
#
# sns.swarmplot(y=delta_dict['sensitivity'], color='r')
# plt.ylabel('ARBA Sensitivity Advantage')
# print(save_fig(size_inches=(3, 4)))
