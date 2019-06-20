import numpy as np

import arba

dim_sbj = 2
dim_img = 2

mu_sbj = np.zeros(dim_sbj)
sig_sbj = np.eye(dim_sbj)
num_sbj = 10

shape = 10, 10, 10
mask = np.zeros(shape).astype(bool)
mask[3:7, 3:7, 3:7] = True

beta = np.random.normal(0, 1, (dim_sbj, dim_img))
beta = np.diag((100, 100))
cov_eps = np.diag(np.random.normal(0, 1, dim_img) ** 2)

# sample sbj features
feat_sbj = np.random.multivariate_normal(mean=mu_sbj, cov=sig_sbj,
                                         size=num_sbj)

# build feat_img (shape0, shape1, shape2, num_sbj, dim_img)
feat_img = np.random.multivariate_normal(mean=np.zeros(dim_img),
                                         cov=cov_eps,
                                         size=(*shape, num_sbj))
# add offset to induce beta
for sbj_idx, offset in enumerate(feat_sbj @ beta):
    feat_img[mask, sbj_idx, :] += offset

# build file_tree
file_tree = arba.data.SynthFileTree.from_array(data=feat_img)
