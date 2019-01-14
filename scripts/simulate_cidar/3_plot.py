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


def make_plots(folder, label):
    f_sg_hist = folder / f'sg_hist{label}.p.gz'
    f_sg_arba = folder / f'sg_arba{label}.p.gz'
    f_effect = folder / 'mask_effect.nii.gz'

    if not f_sg_hist.exists():
        return

    sg_hist = file.load(f_sg_hist)
    sg_arba = file.load(f_sg_arba)

    # resolve space
    sg_hist.space_resolve(sg_hist.tree_history.nodes)
    # todo: load / save sg_hist + sg_arba as same file ...
    # sg_hist.space_resolve(sg_arba.nodes)

    # reg_highlight = set(sg_arba.nodes)
    reg_highlight = set()

    reg_list = None

    if f_effect.exists():
        mask = Mask.from_nii(f_effect)
    else:
        mask = None

    # size vs error
    plot.size_vs_error_normed(sg_hist=sg_hist, n_max=30)
    save_fig(f_out=folder / f'size_vs_error{label}.pdf')

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
    save_fig(f_out=folder / f'size_vs_error_tree{label}.pdf')

    # size v maha
    plot.size_v_mahalanobis(sg=sg_hist.tree_history,
                            reg_list=reg_list,
                            mask=mask,
                            reg_highlight=reg_highlight,
                            dict_highlight={'linewidths': 3,
                                            'edgecolors': 'k'},
                            log_x=True,
                            log_y=True)
    save_fig(f_out=folder / f'size_vs_maha{label}.pdf')

    # size v wmaha
    plot.size_v_wmahalanobis(sg=sg_hist.tree_history,
                             reg_list=reg_list,
                             mask=mask,
                             reg_highlight=reg_highlight,
                             dict_highlight={'linewidths': 3,
                                             'edgecolors': 'k'},
                             log_x=True,
                             log_y=True)
    save_fig(f_out=folder / f'size_vs_wmaha{label}.pdf')

    # size v pval
    plot.size_v_pval(sg=sg_hist.tree_history,
                     mask=mask,
                     reg_highlight=reg_highlight,
                     dict_highlight={'linewidths': 3,
                                     'edgecolors': 'k'},
                     log_x=True,
                     log_y=True)
    save_fig(f_out=folder / f'size_vs_pval{label}.pdf')


if __name__ == '__main__':
    from tqdm import tqdm

    folder = folder / '2019_Jan_14_09_02_41'

    dict_highlight = {'linewidths': 2,
                      'edgecolors': 'k'}
    par_flag = True

    arg_list = []
    s_folder_glob = '*maha*run*'
    for _folder in tqdm(folder.glob(s_folder_glob)):
        arg_list.append({'folder': _folder,
                         'label': ''})
        arg_list.append({'folder': _folder,
                         'label': '_test'})

    if par_flag:
        run_par_fnc(make_plots, arg_list)
    else:
        for d in tqdm(arg_list):
            make_plots(**d)
