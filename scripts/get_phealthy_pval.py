from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from mh_pytools import file
from pnl_data.set.cidar_post import folder
from pnl_segment.space import PointCloud

folder = folder / '2018_Nov_16_12_34AM35'
compute = True
plot = True

spec = .05
n_region_max = 20


def get_pval(reg):
    return reg.pval


if compute:
    # computes max healthy maha from all samples
    snr_c_maha_dict = defaultdict(list)
    snr_c_pval_dict = defaultdict(list)
    s_folder_glob = '*snr*run*'
    for _folder in tqdm(folder.glob(s_folder_glob)):
        f_sg_hist = _folder / 'sg_hist.p.gz'
        f_effect_mask = _folder / 'effect_mask.nii.gz'
        f_effect = _folder / 'effect.p.gz'
        effect = file.load(f_effect)
        sg_hist = file.load(f_sg_hist)
        pc_effect = PointCloud.from_nii(f_effect_mask)
        # sg_arba = sg_hist.cut_spanning_region(get_pval, max=False,
        #                                       n_region_max=n_region_max)
        sg_arba, _, _ = sg_hist.cut_hierarchical(spec=spec)
        snr = effect.snr

        for reg in sg_hist.tree_history.nodes:
            # compute % healthy
            n_union = len(reg.pc_ijk.intersection(pc_effect))
            per_healthy = n_union / len(reg.pc_ijk)

            snr_c_maha_dict[snr].append((per_healthy, reg.maha * len(reg)))
            if reg in sg_arba.nodes:
                snr_c_pval_dict[snr].append((per_healthy, reg.pval))
    file.save(snr_c_maha_dict, folder / 'perc_healthy_wmaha.p.gz')
    file.save(snr_c_pval_dict, folder / 'perc_healthy_pval.p.gz')

if plot:
    sns.set(font_scale=1.2)
    snr_c_maha_dict = file.load(folder / 'healthy_maha.p.gz')
    snr_c_pval_dict = file.load(folder / 'perc_healthy_pval.p.gz')

    f_out = folder / 'snr_vs_wmaha.pdf'
    snr_list = sorted(snr_c_maha_dict.keys())
    with PdfPages(f_out) as pdf:
        for snr in snr_list:
            c = [x[0] for x in snr_c_maha_dict[snr]]
            wmaha = [x[1] for x in snr_c_maha_dict[snr]]

            plt.scatter(c, wmaha, alpha=.2)
            ax = plt.gca()
            ax.set_xlabel('% effect')
            ax.set_ylabel('weighted maha')
            ax.set_yscale('log')
            plt.yticks(np.logspace(0, 5, 6))
            plt.suptitle(f'snr: {snr:.3e}')
            pdf.savefig(plt.gcf())
            plt.gcf().set_size_inches(8, 8)
            plt.close()

    f_out = folder / 'snr_v_perc_healthy_v_pval.pdf'
    with PdfPages(f_out) as pdf:
        for snr in snr_list:
            c = [x[0] for x in snr_c_pval_dict[snr]]
            pval = [x[1] for x in snr_c_pval_dict[snr]]

            plt.scatter(c, pval, alpha=.9)
            ax = plt.gca()
            ax.set_xlabel('% effect')
            ax.set_ylabel('pval')
            ax.set_yscale('log')
            ax.set_xlim(0, 1)
            ax.axhline(spec / n_region_max, color='k')
            plt.yticks(np.logspace(-6, 0, 7))
            plt.suptitle(f'snr: {snr:.3e}')
            pdf.savefig(plt.gcf())
            plt.gcf().set_size_inches(8, 8)
            plt.close()
