import tempfile
from collections import defaultdict
from string import ascii_uppercase

import nibabel as nib
import numpy as np
import pathlib
from .file_tree import FileTree


class SynthFileTree(FileTree):
    """ builds gaussian noise nii in temp location (for testing purposes)

    features are labelled alphabetically ('feat_A', 'feat_B', ....)
    sbj are labelled numerically ('sbj_0', 'sbj_1', ...)
    """

    def __init__(self, n_sbj, shape, mu=0, cov=1, folder=None):
        """

        Args:
            n_sbj (int): number of sbj
            shape (tuple): img shape
            mu (np.array): average feature
            cov (np.array): feature covar
        """
        if folder is None:
            folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        else:
            folder = pathlib.Path(folder)

        mu = np.atleast_1d(mu)
        cov = np.atleast_2d(cov)

        n_feat = len(mu)
        feat_list = [f'feat_{x}' for x in ascii_uppercase[:n_feat]]

        sbj_feat_file_tree = defaultdict(dict)
        for sbj_idx in range(n_sbj):
            # sample img
            x = np.random.multivariate_normal(mean=mu, cov=cov, size=shape)

            # store img (as nii, then in sbj_feat_file_tree)
            sbj = f'sbj_{sbj_idx}'
            for feat_idx, feat in enumerate(feat_list):
                f = tempfile.NamedTemporaryFile(suffix='.nii.gz',
                                                prefix=f'{sbj}_{feat}_').name
                f = folder / pathlib.Path(f).name
                img = nib.Nifti1Image(x[..., feat_idx], affine=np.eye(4))
                img.to_filename(str(f))

                # store
                sbj_feat_file_tree[sbj][feat] = f

        super().__init__(sbj_feat_file_tree=sbj_feat_file_tree)
