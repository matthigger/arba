import networkx as nx
import numpy as np
from tqdm import tqdm

from arba.plot import size_v_pval, save_fig
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

# load
folder = folder / 'arba_cv_MF_FA_test' / 'arba_permute'

f_sg_old = folder / 'sg_old_t2.p.gz'
f_sg_new = folder / 'sg_new_t2.p.gz'


def print_history_min_pval(f_sg, label):
    sg = file.load(f_sg)
    min_pval = np.inf
    for reg in tqdm(sg.nodes):
        if reg.pval < min_pval:
            reg_min = reg
            min_pval = reg.pval
    reg_list = [reg_min] + \
               list(nx.descendants(sg, reg_min)) + \
               list(nx.ancestors(sg, reg_min))
    size_v_pval(sg, min_reg_size=100, reg_list=reg_list)
    print(save_fig(f_out=folder / f'history_min_pval_{label}.pdf'))


print_history_min_pval(f_sg_new, label='new')

print_history_min_pval(f_sg_old, label='old')
