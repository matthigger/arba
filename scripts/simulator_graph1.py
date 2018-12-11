import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from collections import namedtuple

folder_out = folder / 'synth_data'

# load
performance = namedtuple('performance', ('sens', 'spec', 'dice', 'auc'))
f_in = folder_out / 'performance_stats.p.gz'
snr_method_perf_tree = file.load(f_in)

snr_list = sorted(snr_method_perf_tree.keys())
method_list = sorted(snr_method_perf_tree[snr_list[0]].keys())

sns.set(font_scale=1.2)
cm = plt.get_cmap('Set1')
color_dict = {method: cm(idx) for idx, method in enumerate(method_list)}

f_out = folder_out / 'sens_spec_snr.pdf'
with PdfPages(f_out) as pdf:
    for snr in snr_list:
        for method in method_list:
            perf_list = snr_method_perf_tree[snr][method]

            sens = [x.sens for x in perf_list]
            spec = [x.spec for x in perf_list]

            plt.scatter(spec, sens, label=method, alpha=.3,
                        color=color_dict[method])
            # plt.scatter(np.mean(spec), np.mean(sens), marker='s', color='w',
            #             edgecolors=color_dict[method], linewidths=2)

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.suptitle(f'snr: {snr:.2E}')
        plt.xlabel('specificity')
        plt.ylabel('sensitivity')
        plt.xlim((0, 1.05))
        plt.ylim((-.05, 1.05))
        pdf.savefig(plt.gcf())
        plt.gcf().set_size_inches(8, 8)
        plt.close()
