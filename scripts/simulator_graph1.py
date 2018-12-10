import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from pnl_segment import plot

folder_out = folder / '2018_Nov_16_12_34AM35'

# load
f_out = folder_out / 'snr_auc_dice.p.gz'
method_snr_auc_dict, method_snr_dice_tree, method_snr_sens_spec_dict, eff_dict = file.load(
    f_out)

method_list = sorted({x[0] for x in method_snr_sens_spec_dict.keys()})
snr_list = sorted({x[1] for x in method_snr_sens_spec_dict.keys()})

sns.set(font_scale=1.2)
cm = plt.get_cmap('Set1')
color_dict = {method: cm(idx) for idx, method in enumerate(method_list)}

f_out = folder_out / 'sens_spec_snr.pdf'
with PdfPages(f_out) as pdf:
    for snr in snr_list:
        for method in method_list:
            sens_spec = method_snr_sens_spec_dict[method, snr]

            sens = [x[0][0] for x in sens_spec]
            spec = [x[0][1] for x in sens_spec]

            plt.scatter(spec, sens, label=method, alpha=.5,
                        color=color_dict[method])
            plt.scatter(np.mean(spec), np.mean(sens), marker='s', color='w',
                        edgecolors=color_dict[method], linewidths=2)

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

# snr_sens_spec_dict = defaultdict(list)
# for (method, snr), sens_spec_list in method_snr_sens_spec_dict.items():
#     sens_spec = np.squeeze(np.nanmean(sens_spec_list, axis=0))
#     snr_sens_spec_dict[method].append((snr, sens_spec))
#
#
# for method, data_list in snr_sens_spec_dict.items():
#     data_list = sorted(data_list)
#     snr = [x[0] for x in data_list]
#     sens = [x[1][0] for x in data_list]
#     spec = [x[1][1] for x in data_list]
#
#     plt.plot(sens, spec, label=method, color=color_dict[method])
#
# plt.legend()
# plt.xlabel('sensitivity')
# plt.ylabel('specificity')


# get average snr
snr, eff_list = next(iter(eff_dict.items()))
eff_mean_list = [eff.mean for eff in eff_list]
eff_mean = np.mean(eff_mean_list, axis=0) / snr
np.set_printoptions(2, suppress=False)

# plot snr vs dice
f_out = folder_out / 'snr_auc.pdf'
with PdfPages(str(f_out)) as pdf:
    fig, ax = plt.subplots(1, 1)
    plot.line_confidence(method_snr_auc_dict, xlabel='snr', ylabel='auc')
    plt.gca().set_xscale('log')
    plt.suptitle(f'average eff @ snr=1: {eff_mean}')
    plt.gcf().set_size_inches(10, 7)
    pdf.savefig(plt.gcf())
    plt.close()

# plot snr vs dice
f_out = folder_out / 'snr_dice.pdf'
with PdfPages(str(f_out)) as pdf:
    fig, ax = plt.subplots(1, 1)
    plot.line_confidence(method_snr_dice_tree, xlabel='snr', ylabel='dice')
    plt.gca().set_xscale('log')
    plt.suptitle(f'average eff @ snr=1: {eff_mean}')
    plt.gcf().set_size_inches(10, 7)
    pdf.savefig(plt.gcf())
    plt.close()
