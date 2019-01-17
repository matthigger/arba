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
    f_effect = folder / 'image' / 'mask_effect.nii.gz'

    if not f_sg_hist.exists():
        return

    sg_hist = file.load(f_sg_hist)
    sg_arba = file.load(f_sg_arba)

    region_spaces = list(reg.pc_ijk for reg in sg_arba.nodes)
    _, tree_hist_resolved = sg_hist.resolve_hist()
    reg_highlight = [reg for reg in tree_hist_resolved.nodes
                     if reg.pc_ijk in region_spaces]

    if f_effect.exists():
        mask = Mask.from_nii(f_effect)
    else:
        mask = None

    # size v pval
    plot.size_v_pval(sg=tree_hist_resolved,
                     mask=mask,
                     reg_highlight=reg_highlight,
                     dict_highlight={'linewidths': 3,
                                     'edgecolors': 'k'},
                     log_x=True,
                     log_y=True)
    save_fig(f_out=folder / f'size_vs_pval{label}.pdf')


if __name__ == '__main__':
    from tqdm import tqdm

    folder = folder / '2019_Jan_17_09_37_26'

    dict_highlight = {'linewidths': 2,
                      'edgecolors': 'k'}
    par_flag = True

    arg_list = []
    s_folder_glob = '*maha*run*'
    for _folder in tqdm(folder.glob(s_folder_glob)):
        arg_list.append({'folder': _folder,
                         'label': '_test'})
        arg_list.append({'folder': _folder,
                         'label': '_seg'})

    if par_flag:
        run_par_fnc(make_plots, arg_list)
    else:
        for d in tqdm(arg_list):
            make_plots(**d)
