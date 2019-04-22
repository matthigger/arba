import os
import pathlib
import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np

from arba.file_tree import FileTree
from arba.region import FeatStat
from arba.space import RefSpace, get_ref


class Model:
    """ given FeatStat per voxel and size, generates images

    only supports scalar feature sets
    """

    @staticmethod
    def from_arrays(array_list):
        ijk_sample_dict = defaultdict(list)
        shape = next(iter(array_list)).shape

        # aggregate per voxel
        for x in array_list:
            if x.shape != shape:
                raise AttributeError('mismatched shape')

            for ijk, _x in np.ndenumerate(x):
                ijk_sample_dict[ijk].append(_x)

        # estimate normal
        ijk_fs_dict = {ijk: FeatStat.from_iter(x_list)
                       for ijk, x_list in ijk_sample_dict.items()}

        return Model(ijk_fs_dict, shape=shape)

    def __init__(self, ijk_fs_dict, shape=None):
        self.ijk_fs_dict = ijk_fs_dict

        self.shape = shape
        if self.shape is None:
            self.shape = tuple(max(ijk[x] for ijk in self.ijk_fs_dict.keys())
                               for x in enumerate(3))

    def combine(self, ijk_list):
        fs = sum(self.ijk_fs_dict[ijk] for ijk in ijk_list)

        for ijk in ijk_list:
            self.ijk_fs_dict[ijk] = fs

    def combine_by_seg(self, seg):

        for val in np.unique(seg.flatten()):
            ijk_list = np.vstack(np.where(seg == val)).T
            ijk_list = [tuple(x) for x in ijk_list]
            self.combine(ijk_list)

    def sample_array(self):
        x = np.ones(self.shape) * np.nan
        for ijk, fs in self.ijk_fs_dict.items():
            x[ijk] = fs.to_normal(bessel=False).rvs()
        return x

    def sample_nii(self, f_out=None, ref=None):
        if f_out is None:
            f, f_out = tempfile.mkstemp(suffix='.nii.gz')
            os.close(f)

        if ref is None:
            ref = RefSpace(affine=np.eye(4), shape=self.shape)
        else:
            ref = get_ref(ref)

        # build image
        x = self.sample_array()

        # save to file
        img = nib.Nifti1Image(x, affine=ref.affine)
        img.to_filename(str(f_out))

        return f_out

    def to_file_tree(self, n, folder=None, label='', **kwargs):
        sbj_feat_file_tree = defaultdict(dict)
        folder = pathlib.Path(folder)
        folder.mkdir(exist_ok=True, parents=True)

        for idx in range(n):
            if folder:
                f_out = folder / f'{label}{idx}.nii.gz'
            else:
                f_out = None
            sbj_feat_file_tree[f'sbj_{idx}']['dummy_feat'] = self.sample_nii(
                f_out=f_out, **kwargs)
        return FileTree(sbj_feat_file_tree)
