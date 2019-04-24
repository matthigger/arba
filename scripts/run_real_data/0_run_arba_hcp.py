from collections import defaultdict

from arba.data import FileTree, Split
from arba.permute import PermuteARBA
from pnl_data.set.hcp_100 import folder, people

from arba.plot import size_v_pval, size_v_t2, size_v_delta, save_fig, size_v_sig_sbj, size_v_sig_space

# arba params
alpha = .05
feat_tuple = 'FA',
num_perm = 100

# run params
par_flag = True
verbose = True

# output folder
s_feat = '-'.join(feat_tuple)
folder_out = folder / f'arba_cv_MF_{s_feat}_new_2'
folder_out.mkdir(exist_ok=True)

folder_data = folder / 'to_100307_low_res'

# get files per feature
sbj_feat_file_tree = defaultdict(dict)
split = defaultdict(list)
for p in people:
    for feat in feat_tuple:
        f = folder_data / f'{p.name}_{feat}.nii.gz'
        if not f.exists():
            raise FileNotFoundError(f)
        sbj_feat_file_tree[p][feat] = f
    split[p.gender].append(p)

# build file tree
file_tree = FileTree(sbj_feat_file_tree)
Split.fix_order(file_tree.sbj_list)
split = Split(split)

# build file tree dict
folder_arba = folder_out / 'arba_permute'
folder_tfce = folder_out / 'tfce'

with file_tree.loaded():
    permute_arba = PermuteARBA(file_tree)

    sg_hist = permute_arba.run_split(split, full_t2=True, verbose=True)

    tree_hist, _ = sg_hist.merge_record.resolve_hist(file_tree, split)

    for fnc_plot in (size_v_pval,
                     size_v_t2,
                     size_v_delta,
                     size_v_sig_sbj,
                     size_v_sig_space):
        fnc_plot(sg=tree_hist, min_reg_size=20)
        save_fig(f_out=folder_out / f'{fnc_plot.__name__}.pdf')

    # permute_arba.run(split, par_flag=par_flag, verbose=True,
    #                  folder=folder_arba, n=num_perm, print_image=True,
    #                  save_self=True, print_tree=True, print_hist=True,
    #                  print_region=True)

    # permute_tfce = PermuteTFCE(file_tree)
    # permute_tfce.run(split, par_flag=True, verbose=True,
    #                  folder=folder_tfce, n=num_perm, print_image=True,
    #                  save_self=True, print_hist=True, print_region=True)
