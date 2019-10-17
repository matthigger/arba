from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import arba


class Performance:
    """ throwaway: tracks sensitivity and specificty per method """

    def __init__(self, stat_label):
        self.stat_label = stat_label
        self.method_stat_ss_list_dict = defaultdict(list)

    def check_in(self, perm_reg, stat):
        for mode, est_mask in perm_reg.mode_est_mask_dict.items():
            sens, spec = arba.effect.get_sens_spec(target=perm_reg.mask_target,
                                                   estimate=est_mask,
                                                   mask=perm_reg.data_img.mask)
            self.method_stat_ss_list_dict[mode].append((stat, sens, spec))

    def plot(self, folder):
        sns.set(font_scale=1)
        fig, ax = plt.subplots(1, 2)
        for idx, feat in enumerate(('sensitivity', 'specificity')):
            plt.sca(ax[idx])
            for method, stat_ss_list in self.method_stat_ss_list_dict.items():
                stat_list = [x[0] for x in stat_ss_list]
                vals = [x[1 + idx] for x in stat_ss_list]
                plt.scatter(stat_list, vals, label=method, alpha=.4)

                d = defaultdict(list)
                for _stat, val in zip(stat_list, vals):
                    d[_stat].append(val)

                x = []
                y = []
                for _stat, val_list in sorted(d.items()):
                    x.append(_stat)
                    y.append(np.mean(val_list))
                plt.plot(x, y)

            plt.ylabel(feat)
            plt.xlabel(self.stat_label)
            plt.legend()
            plt.gca().set_xscale('log')
        f = folder / f'{self.stat_label}_vs_sens_spec.pdf'
        arba.plot.save_fig(f, size_inches=(10, 4))

        print(folder)
