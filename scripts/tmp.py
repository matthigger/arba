import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from mh_pytools import file
from mh_pytools.parallel import run_par_fnc
from pnl_data.set.cidar_post import folder
from pnl_segment import plot
from pnl_segment.space import Mask


def get_pval(reg):
    return reg.pval


def make_plots(folder, spec, n_region_max):
    f_sg_hist = folder / 'sg_hist.p.gz'
    f_effect = folder / 'effect_mask.nii.gz'

    if not f_sg_hist.exists():
        return

    sg_hist = file.load(f_sg_hist)
    size = sum(len(r) for r in sg_hist.root_iter)
    s = np.linspace(1, size)
    pval_thresh = (s / size) * spec

    # sg_span, _, _ = sg_hist.cut_hierarchical(spec)
    sg_span = sg_hist.cut_p_error_each_step(.2)
    print(f'sg_span has {len(sg_span)} regions')
    reg_highlight = set(sg_span.nodes)

    if f_effect.exists():
        mask = Mask.from_nii(f_effect)
        if plot_all:
            reg_list = None
        else:
            reg_list = [reg for reg in sg_hist.tree_history.nodes if
                        any(mask[ijk] for ijk in reg.pc_ijk)]
    else:
        mask = None
        reg_list = None

    f_out = folder / 'size_vs_error.pdf'
    with PdfPages(str(f_out)) as pdf:
        plot.size_vs_error_normed(sg_hist=sg_hist, n_max=30)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

        _reg_list = set()
        for sg in list(sg_hist)[-20:]:
            _reg_list |= set(sg.nodes)

        plot.size_v_error(sg=sg_hist.tree_history,
                          mask=mask,
                          reg_list=_reg_list,
                          reg_highlight=reg_highlight,
                          dict_highlight={'linewidths': 3,
                                          'edgecolors': 'k'},
                          log_x=True,
                          log_y=True)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

    # f_out = folder / 'size_vs_maha.pdf'
    # with PdfPages(str(f_out)) as pdf:
    #     plot.size_v_mahalanobis(sg=sg_hist.tree_history,
    #                             reg_list=reg_list,
    #                             mask=mask,
    #                             reg_highlight=reg_highlight,
    #                             dict_highlight={'linewidths': 3,
    #                                             'edgecolors': 'k'},
    #                             log_x=True,
    #                             log_y=True)
    #
    #     fig = plt.gcf()
    #     fig.set_size_inches(10, 7)
    #     pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    #     plt.close()
    #
    # f_out = folder / 'size_vs_wmaha.pdf'
    # with PdfPages(str(f_out)) as pdf:
    #     plot.size_v_wmahalanobis(sg=sg_hist.tree_history,
    #                              reg_list=reg_list,
    #                              mask=mask,
    #                              reg_highlight=reg_highlight,
    #                              dict_highlight={'linewidths': 3,
    #                                              'edgecolors': 'k'},
    #                              log_x=True,
    #                              log_y=True)
    #
    #     fig = plt.gcf()
    #     fig.set_size_inches(10, 7)
    #     pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # f_out = folder / 'size_vs_pval_mono.pdf'
    # with PdfPages(str(f_out)) as pdf:
    #     plt.plot(s, pval_thresh, color='g')
    #     ax = plt.gca()
    #     plot.size_v_pval(sg=sg_hist.tree_history,
    #                      reg_list=reg_list,
    #                      mask=mask,
    #                      reg_highlight=reg_highlight,
    #                      dict_highlight={'linewidths': 3,
    #                                      'edgecolors': 'k'},
    #                      log_x=True,
    #                      log_y=True,
    #                      ax=ax)
    #
    #     fig = plt.gcf()
    #     fig.set_size_inches(10, 7)
    #     pdf.savefig(fig,  bbox_inches='tight', pad_inches=0)
    #     plt.close()

    sg_hist = file.load(f_sg_hist)
    f_out = folder / 'size_vs_pval.pdf'
    with PdfPages(str(f_out)) as pdf:
        plt.plot(s, pval_thresh, color='g')
        ax = plt.gca()
        plot.size_v_pval(sg=sg_hist.tree_history,
                         mask=mask,
                         reg_highlight=reg_highlight,
                         dict_highlight={'linewidths': 3,
                                         'edgecolors': 'k'},
                         log_x=True,
                         log_y=True,
                         ax=ax)

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()


folder = folder / 'synth_data'
dict_highlight = {'linewidths': 2,
                  'edgecolors': 'k'}
plot_all = True
par_flag = True
spec = .01
n_region_max = 15

arg_list = []
s_folder_glob = '*snr*run*'
# s_folder_glob = 'snr_1.000E+02_run000'
for _folder in tqdm(folder.glob(s_folder_glob)):
    if par_flag:
        arg_list.append({'folder': _folder,
                         'spec': spec,
                         'n_region_max': n_region_max})
    else:
        make_plots(_folder, spec, n_region_max)

if par_flag:
    run_par_fnc(make_plots, arg_list)

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set(font_scale=1.2)
# maha_dm = [r.maha * len(r) for r in sg_hist_dm.tree_history.nodes]
# maha = [r.maha * len(r) for r in sg_hist.tree_history.nodes]
#
# bins = np.linspace(0, 50, 100)
# plt.hist(maha, label=f'snr={effect.snr}', alpha=.4, bins=bins)
# plt.hist(maha_dm, label='dummy', alpha=.4, bins=bins)
# plt.legend()
# ax = plt.gca()
# ax.set_yscale('log')
# ax.set_xlabel('maha * reg_size')
# ax.set_ylabel('count')
# plt.show()
