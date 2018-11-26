import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np

from ..seg_graph import FileTree
from ..space import RefSpace, get_ref


class ImageGen:
    """ given normal distribution per voxel, generates images + file trees

    only supports scalar feature sets
    """

    def __init__(self, ijk_fs_dict, shape=None):
        self.ijk_fs_dict = ijk_fs_dict
        self.ijk_norm_dict = {ijk: fs.to_normal()
                              for ijk, fs in ijk_fs_dict.items()}

        self.shape = shape
        if self.shape is None:
            self.shape = tuple(max(ijk[x] for ijk in self.ijk_fs_dict.keys())
                               for x in enumerate(3))

    def sample(self, f_out=None, ref=None):
        if f_out is None:
            _, f_out = tempfile.mkstemp(suffix='.nii.gz')

        if ref is None:
            ref = RefSpace(affine=np.eye(4), shape=self.shape)
        else:
            ref = get_ref(ref)

        # build image
        x = np.ones(self.shape) * np.nan
        for ijk, norm in self.ijk_norm_dict.items():
            x[ijk] = norm.rvs()

        # save to file
        img = nib.Nifti1Image(x, affine=ref.affine)
        img.to_filename(str(f_out))

        return f_out

    def get_file_tree(self, n):
        sbj_feat_file_tree = defaultdict(dict)
        for idx in range(n):
            sbj_feat_file_tree[f'sbj_{idx}']['dummy_feat'] = self.sample()
        return FileTree(sbj_feat_file_tree)
