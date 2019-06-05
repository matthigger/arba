import pathlib
import tempfile

import nibabel as nib
import numpy as np

import arba
from mh_pytools import file, parallel

# file tree
n_sbj = 10
mu = (0, 0)
cov = np.eye(2)
shape = 5, 5, 1
t
# effect
mask = np.zeros(shape)
mask[2:4, 2:4, 0] = 1
offset = (1, 1)

# bayes threshold
alpha = .05

# build file tree
folder = pathlib.Path(tempfile.TemporaryDirectory().name)
print(folder)
_folder = folder / 'data_orig'
_folder.mkdir(exist_ok=True, parents=True)
ft = arba.data.SynthFileTree(n_sbj=n_sbj, shape=shape, mu=mu, cov=cov,
                             folder=_folder)

# build effect
effect = arba.simulate.Effect(mask=mask, offset=offset)

# build 'split', defines sbj which are affected
arba.data.Split.fix_order(ft.sbj_list)
half = int(n_sbj / 2)
split = arba.data.Split({False: ft.sbj_list[:half],
                         True: ft.sbj_list[half:]})

# go
with ft.loaded(split_eff_list=[(split, effect)]):
    # write features to block img
    for feat_idx, feat in enumerate(ft.feat_list):
        img = nib.Nifti1Image(ft.data[:, :, :, :, feat_idx], ft.ref.affine)
        img.to_filename(str(folder / f'{feat}.nii.gz'))

    # agglomerative clustering
    sg_hist = arba.seg_graph.SegGraphHistory(file_tree=ft, split=split)
    sg_hist.reduce_to(1)

    # save
    f = folder / 'sg_hist.p.gz'
    file.save(sg_hist, f)

    # run bayes on each volume, record mv_norm and lower_bnd
    arg_list = list()
    merge_record = sg_hist.merge_record
    for node in merge_record.nodes:
        # get mask
        pc = merge_record.get_pc(node)
        mask = pc.to_mask()

        #
        data0, data1 = arba.bayes.get_data(file_tree=ft, mask=mask,
                                           split=split)
        d = {'data0': data0,
             'data1': data1,
             'node': node}

        arg_list.append(d)


def fnc(data0, data1, node):
    # run bayes
    _, trace = arba.bayes.estimate_delta(data0=data0, data1=data1, cores=1)

    x = trace['delta']
    mu = np.mean(x, axis=0)
    cov = np.cov(x.T)

    return node, (mu, cov)


# run par func
res_out = parallel.run_par_fnc(fnc, arg_list, desc='bayes per region')
node_mu_cov_dict = dict(res_out)

# save
f = folder / 'node_mu_cov_dict.p.gz'
file.save(node_mu_cov_dict, f)
