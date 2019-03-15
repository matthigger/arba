import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def scatter_topo_cov_det(tree_hist):
    ylabel = 'det of pooled covar'

    def fnc(reg):
        fs0, fs1 = reg.fs_dict.values()
        cov = (fs0.n * fs0.cov +
               fs1.n * fs1.cov) / (fs0.n + fs1.n)
        return np.linalg.det(cov)

    return scatter_topo(tree_hist, fnc, ylabel)


def scatter_topo(tree_hist, fnc, ylabel, log_y=False):
    # assign one edge of every edge pair as 'positive'
    positive_edge_set = set()
    for r in tree_hist.nodes:
        r_child = next(iter(tree_hist.predecessors(r)), None)
        if r_child is None:
            continue
        positive_edge_set.add((r_child, r))

    bin_str_dict = dict()
    for r_orig in tree_hist.nodes:
        if len(r_orig) > 1:
            continue
        bin_str = ''
        r = r_orig
        while True:
            list_neigh = list(tree_hist.successors(r))
            if not list_neigh:
                break
            if (r, list_neigh[0]) in positive_edge_set:
                bin_str += '1'
            else:
                bin_str += '0'
            r = list_neigh[0]

        if bin_str:
            bin_str_dict[r_orig] = bin_str

    # pos = graphviz_layout(tree_hist, prog='dot')
    # nx.draw(tree_hist,pos=pos)

    n = max(len(v) for v in bin_str_dict.values())
    x_val = {r: int(s.zfill(n)[::-1], base=2) for r, s in bin_str_dict.items()}
    x_val = {r: idx for idx, r in
             enumerate(sorted(x_val.keys(), key=lambda r: x_val[r]))}

    for r in sorted(tree_hist.nodes, key=len):
        if r in x_val.keys():
            continue
        x_val[r] = max([x_val[_r] for _r in tree_hist.predecessors(r)])

    node_pos = {r: (x_val[r], fnc(r)) for r in tree_hist.nodes}
    nx.draw_networkx_nodes(tree_hist.nodes, pos=node_pos, node_size=1,
                           node_color='k')
    nx.draw_networkx_edges(tree_hist, pos=node_pos, arrows=False)

    if log_y:
        plt.gca().set_yscale('log')

    plt.ylabel(ylabel)
    plt.grid(True)


if __name__ == '__main__':
    from mh_pytools import file

    f = '/home/mu584/dropbox_pnl/data/hcp_100/vox_sbj_72_FA-MD_min_var/t2_4_1.0E+00_effect0/arba_cv/save/sg_hist_seg.p.gz'
    sg_hist_seg = file.load(f)

    tree_hist, _ = sg_hist_seg.resolve_hist()
    # scatter_topo_cov_det(tree_hist)
    def fnc(reg):
        fs0, fs1 = reg.fs_dict.values()
        cov = (fs0.n * fs0.cov +
               fs1.n * fs1.cov) / (fs0.n + fs1.n)
        return np.linalg.det(cov)

    node_pos = {r: (len(r) * r.t2 / fnc(r), r.pval) for r in tree_hist.nodes}
    nx.draw_networkx_nodes(tree_hist.nodes, pos=node_pos, node_size=1,
                           node_color='r')
    nx.draw_networkx_edges(tree_hist, pos=node_pos, arrows=False)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.ylabel('pval')
    plt.grid(True)
    pass
