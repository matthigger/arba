import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from arba import plot
from arba.space import Mask
from mh_pytools import file
from mh_pytools.parallel import run_par_fnc


def save_fig(f_out):
    fig = plt.gcf()
    fig.set_size_inches(14, 7)

    with PdfPages(str(f_out)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close()


def make_plots(folder, label):
    f_sg_hist = folder / f'sg_hist{label}.p.gz'
    f_sg_arba = folder / f'sg_arba{label}.p.gz'
    folder_image = folder.parent
    f_effect = folder_image / 'mask_effect_EFFECT.nii.gz'

    if not f_sg_hist.exists():
        return

    sg_hist = file.load(f_sg_hist)
    sg_arba = file.load(f_sg_arba)

    region_spaces = list(reg.pc_ijk for reg in sg_arba.nodes)
    tree_hist, _ = sg_hist.merge_record.resolve_hist()
    reg_highlight = [reg for reg in tree_hist.nodes
                     if reg.pc_ijk in region_spaces]

    if f_effect.exists():
        mask = Mask.from_nii(f_effect)
    else:
        mask = None

    # size v pval
    fig, ax = plt.subplots(1, 2)
    plot.size_v_pval(sg=tree_hist,
                     mask=mask,
                     reg_highlight=reg_highlight,
                     dict_highlight={'linewidths': 3,
                                     'edgecolors': 'k'},
                     log_x=True,
                     log_y=True,
                     min_reg_size=5,
                     ax=ax[0])
    plot.size_v_wt2(sg=tree_hist,
                    mask=mask,
                    reg_highlight=reg_highlight,
                    dict_highlight={'linewidths': 3,
                                    'edgecolors': 'k'},
                    log_x=True,
                    log_y=True,
                     min_reg_size=5,
                    ax=ax[1])
    save_fig(f_out=folder_image / f'size_vs_x_{label}.pdf')


if __name__ == '__main__':
    from tqdm import tqdm

    from pnl_data.set.hcp_100 import folder

    folder = folder / 'vox_sbj_72_FA_min_var'

    dict_highlight = {'linewidths': 2,
                      'edgecolors': 'k'}
    par_flag = True

    arg_list = []
    s_folder_glob = '*t2*effect0'
    for _folder in folder.glob(s_folder_glob):
        for _file in _folder.glob('**/save/sg_hist*'):
            __folder = _file.parent
            label = _file.name.split('.')[0].replace('sg_hist', '')
            if 'seg' not in label:
                continue
            arg_list.append({'folder': __folder,
                             'label': label})

    if par_flag:
        run_par_fnc(make_plots, arg_list)
    else:
        for d in tqdm(arg_list):
            make_plots(**d)
