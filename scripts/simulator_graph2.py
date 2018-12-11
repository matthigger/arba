import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file
from mh_pytools.parallel import run_par_fnc
from pnl_data.set.cidar_post import folder
from pnl_segment import plot
from pnl_segment.space import Mask


def save_fig(f_out):
    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    with PdfPages(str(f_out)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close()


def make_plots(folder):
    f_sg_hist = folder / 'sg_hist.p.gz'
    f_sg_arba = folder / 'sg_arba.p.gz'
    f_effect = folder / 'effect_mask.nii.gz'

    if not f_sg_hist.exists():
        return

    sg_hist = file.load(f_sg_hist)
    sg_arba = file.load(f_sg_arba)

    reg_highlight = set(sg_arba.nodes)

    reg_list = None

    if f_effect.exists():
        mask = Mask.from_nii(f_effect)
    else:
        mask = None

    # size vs error
    plot.size_vs_error_normed(sg_hist=sg_hist, n_max=30)
    save_fig(f_out=folder / 'size_vs_error.pdf')

    # size vs error tree
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
    save_fig(f_out=folder / 'size_vs_error_tree.pdf')

    # size v maha
    plot.size_v_mahalanobis(sg=sg_hist.tree_history,
                            reg_list=reg_list,
                            mask=mask,
                            reg_highlight=reg_highlight,
                            dict_highlight={'linewidths': 3,
                                            'edgecolors': 'k'},
                            log_x=True,
                            log_y=True)
    save_fig(f_out=folder / 'size_vs_maha.pdf')

    # size v wmaha
    plot.size_v_wmahalanobis(sg=sg_hist.tree_history,
                             reg_list=reg_list,
                             mask=mask,
                             reg_highlight=reg_highlight,
                             dict_highlight={'linewidths': 3,
                                             'edgecolors': 'k'},
                             log_x=True,
                             log_y=True)
    save_fig(f_out=folder / 'size_vs_wmaha.pdf')

    # size v pval
    sg_hist = file.load(f_sg_hist)
    plot.size_v_pval(sg=sg_hist.tree_history,
                     mask=mask,
                     reg_highlight=reg_highlight,
                     dict_highlight={'linewidths': 3,
                                     'edgecolors': 'k'},
                     log_x=True,
                     log_y=True)
    save_fig(f_out=folder / 'size_vs_pval.pdf')


if __name__ == '__main__':
    from tqdm import tqdm

    folder = folder / 'synth_data'
    dict_highlight = {'linewidths': 2,
                      'edgecolors': 'k'}
    par_flag = True

    arg_list = []
    s_folder_glob = '*snr*run*'
    for _folder in tqdm(folder.glob(s_folder_glob)):
        arg_list.append({'folder': _folder})

    if par_flag:
        run_par_fnc(make_plots, arg_list)
    else:
        for d in tqdm(arg_list):
            make_plots(**d)
