import pathlib
import tempfile
from collections import defaultdict
from string import ascii_uppercase

import nibabel as nib
import numpy as np

from .file_tree import FileTree


class SynthFileTree(FileTree):
    """ builds gaussian noise nii in temp location (for testing purposes)

    features are labelled alphabetically ('img_featA', 'img_featB', ....)
    sbj are labelled numerically ('sbj0', 'sbj1', ...)
    """

    @staticmethod
    def get_sbj_list(n):
        return [f'sbj{idx}' for idx in range(n)]

    @staticmethod
    def get_feat_list(n):
        return [f'img_feat{x}' for x in ascii_uppercase[:n]]

    @classmethod
    def from_array(cls, data, folder=None):
        """ given a data matrix, writes files to nii and returns FileTree

        Args:
            data (np.array): (shape0, shape1, shape2, num_sbj, num_feat)
            folder: location to store output files

        Returns:
            file_tree (SynthFileTree)
        """

        if folder is None:
            folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        else:
            folder = pathlib.Path(folder)
        if not folder.exists():
            folder.mkdir(exist_ok=True, parents=True)

        assert len(data.shape) == 5, 'data must be of dimension 5'

        num_sbj, num_feat = data.shape[3:]

        sbj_list = cls.get_sbj_list(num_sbj)

        sbj_feat_file_tree = defaultdict(dict)
        for sbj_idx, sbj in enumerate(sbj_list):
            for feat_idx, feat in enumerate(cls.get_feat_list(num_feat)):
                # write img to file
                f = folder / f'{sbj}_{feat}.nii.gz'
                x = data[:, :, :, sbj_idx, feat_idx]
                img = nib.Nifti1Image(x, affine=np.eye(4))
                img.to_filename(str(f))

                # store
                sbj_feat_file_tree[sbj][feat] = f

        # todo: type should be SynthFileTree ... but this is classmethod?
        return FileTree(sbj_feat_file_tree=sbj_feat_file_tree,
                        sbj_list=sbj_list)

    def __init__(self, num_sbj, shape, mu=0, cov=1, folder=None):
        """

        Args:
            num_sbj (int): number of sbj
            shape (tuple): img shape
            mu (np.array): average feature
            cov (np.array): feature covar
        """
        if folder is None:
            folder = pathlib.Path(tempfile.TemporaryDirectory().name)
        else:
            folder = pathlib.Path(folder)
            folder.mkdir(exist_ok=True, parents=True)

        mu = np.atleast_1d(mu)
        cov = np.atleast_2d(cov)

        feat_list = self.get_feat_list(len(mu))

        sbj_feat_file_tree = defaultdict(dict)
        for sbj in self.get_sbj_list(num_sbj):
            # sample img
            x = np.random.multivariate_normal(mean=mu, cov=cov, size=shape)

            # store img (as nii, then in sbj_feat_file_tree)
            for feat_idx, feat in enumerate(feat_list):
                f = tempfile.NamedTemporaryFile(suffix='.nii.gz',
                                                prefix=f'{sbj}_{feat}_').name
                f = folder / pathlib.Path(f).name
                img = nib.Nifti1Image(x[..., feat_idx], affine=np.eye(4))
                img.to_filename(str(f))

                # store
                sbj_feat_file_tree[sbj][feat] = f

        super().__init__(sbj_feat_file_tree=sbj_feat_file_tree)
