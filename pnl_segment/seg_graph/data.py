import random
from collections import defaultdict

import nibabel as nib
import numpy as np
from tqdm import tqdm

from ..region.feat_stat import FeatStatEmpty, FeatStat
from ..space import get_ref, Mask, PointCloud


class FileTree:
    """ manages scalar files of 3d volumes

    accessed as: file_tree[sbj][feat] = '/path/to/file.nii.gz'

    we store ijk_fs_dict, as its comparably light on memory and allows one to
    avoid loading the data again

    Attributes:
        sbj_feat_file_tree (tree): tree, keys are sbj, feature, leafs are files
        feat_list (list): list of features, defines ordering of feat
        ref (RefSpace): defines shape and affine of data
        ijk_fs_dict (dict): keys are ijk tuple, values are FeatStat objs
    """

    @property
    def point_cloud(self):
        if self._point_cloud is None:
            self._point_cloud = self.get_point_cloud()

        return self._point_cloud

    @property
    def sbj_iter(self):
        return iter(self.sbj_feat_file_tree.keys())

    def resample_iid(self, mask):
        """ resamples values in self.ijk_fs_dict to be iid

        samples are drawn from normal(mu, cov) where mu, cov are the mean and
        sample variance of all voxels in mask from self.ijk_fs_dict

        Args:
            mask (Mask): defines values to resample

        Returns:
            file_tree (FileTree): copy of file tree with ONLY these voxels,
                                  which have been resampled
        """
        # build normal distribution with same 1st, 2nd moments as vox in mask
        pc = PointCloud.from_mask(mask)
        pc &= set(self.ijk_fs_dict.keys())
        fs = sum(self.ijk_fs_dict[ijk] for ijk in pc)
        fs_normal = fs.to_normal()

        # resample
        for ijk in pc:
            # sample same number of values
            n = self.ijk_fs_dict[ijk].n
            x = np.atleast_2d(fs_normal.rvs(n)).T

            # build new feat_stat and store
            fs = FeatStat.from_array(x)
            self.ijk_fs_dict[ijk] = fs

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree):
        # todo: all attributes should be internal
        # init
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.ijk_fs_dict = defaultdict(FeatStatEmpty)
        self._point_cloud = None

        # assign and validate feat_list
        self.feat_list = None
        for sbj in self.sbj_iter:
            feat_list = sorted(self.sbj_feat_file_tree[sbj].keys())
            if self.feat_list is None:
                self.feat_list = feat_list
            elif self.feat_list != feat_list:
                raise AttributeError('feat_list mismatch')

        # assign and validate ref space
        self.ref = None
        for f_nii_dict in self.sbj_feat_file_tree.values():
            for f_nii in f_nii_dict.values():
                if self.ref is None:
                    self.ref = get_ref(f_nii)
                elif self.ref != get_ref(f_nii):
                    raise AttributeError('space mismatch')

    def load(self, ijk_set=None, verbose=False):
        """ loads files, adds data to statistics per voxel

        Args:
            ijk_set (Set): ijk tuples to load
            verbose (bool): toggles command line output
        """

        if ijk_set is None:
            ijk_set = self.point_cloud

        # store files
        tqdm_dict = {'disable': not verbose,
                     'desc': 'load sbj'}
        for sbj, f_nii_dict in tqdm(self.sbj_feat_file_tree.items(),
                                    **tqdm_dict):
            # get data
            ijk_data_dict = self.get_ijk_fs_dict(sbj, ijk_set=ijk_set)

            # update stats per sbj
            for ijk, fs in ijk_data_dict.items():
                # store aggregate stats
                self.ijk_fs_dict[ijk] += fs

        # we 'fix' the dict (not defaultdict) prevent errors once loaded
        self.ijk_fs_dict = dict(self.ijk_fs_dict)

    def get_ijk_fs_dict(self, sbj, ijk_set):
        """ returns data per ijk, loads from original files

        Args:
            sbj: sbj to load
            ijk_set (set): restrict ijk values in return dict

        Returns:
            ijk_fs_dict (dict): keys are ijk tuple, values are array of data
                                  with shape (n, len(self.feat_list))
        """

        # load data
        f_nii_list = [self.sbj_feat_file_tree[sbj][feat]
                      for feat in self.feat_list]
        data_list = [nib.load(str(f)).get_data() for f in f_nii_list]

        # store data
        ijk_fs_dict = dict()
        for ijk in ijk_set:
            x = np.array([data[ijk] for data in data_list])
            if not x.all():
                continue
            ijk_fs_dict[ijk] = FeatStat.from_array(x, obs_greater_dim=False)

        return ijk_fs_dict

    def apply_mask(self, mask):
        ft = FileTree(self.sbj_feat_file_tree)
        ft.ijk_fs_dict = defaultdict(FeatStatEmpty)

        ijk_set = PointCloud.from_mask(mask)
        ijk_set &= {x for x in self.ijk_fs_dict.keys()}
        for ijk in ijk_set:
            ft.ijk_fs_dict[ijk] = self.ijk_fs_dict[ijk]

        return ft

    def to_array(self, fnc=None, feat=None):
        """ returns array built from feat stat at each ijk

        either fnc xor feat, if fnc passed it is applied to each feat_stat, if
        feat is passed then the mean feature at each ijk is returned

        Args:
            fnc (fnc): to be applied
            feat (str): feature to get mean of

        Returns:
            x (np.array):
        """
        x = np.zeros(self.ref.shape)

        if (fnc is None) == (feat is None):
            raise AttributeError('either fnc xor feat required')

        if feat is not None:
            feat_idx = self.feat_list.index(feat)

            def fnc(reg):
                return reg.mu.flatten()[feat_idx]

        for ijk, reg in self.ijk_fs_dict.items():
            x[ijk] = fnc(reg)

        return x

    def to_nii(self, f_out, **kwargs):
        x = self.to_array(**kwargs)
        img = nib.Nifti1Image(x, self.ref.affine)
        img.to_filename(str(f_out))

    def get_point_cloud(self):
        """ returns a point_cloud which has all voxels with complete data
        """
        # get set of all voxels present in all data file
        f_nii_list = list()
        for sbj in self.sbj_iter:
            f_nii_list += list(self.sbj_feat_file_tree[sbj].values())
        pc_list = [PointCloud.from_nii(f_nii) for f_nii in f_nii_list]
        pc = PointCloud(set.intersection(*pc_list), ref=self.ref)

        return pc

    def get_subset(self, sbj_list, **kwargs):
        # init new file tree obj from only sbj in sbj_list
        sbj_feat_file_tree = {sbj: self.sbj_feat_file_tree[sbj]
                              for sbj in sbj_list}
        ft = FileTree(sbj_feat_file_tree)
        ft.load(**kwargs)
        return ft

    def split(self, p=.5, verbose=False, **kwargs):
        """ splits data into two groups

        Args:
            p (float): in (0, 1), splits data to have at least p percent of sbj
                       in the first grp
            verbose (bool): toggles command line output

        Returns:
            file_tree_list (list): FileTree for each grp
        """

        # compute n_effect
        n_grp0 = np.ceil(p * len(self)).astype(int)

        # split sbj into health and effect groups
        sbj_set = set(self.sbj_iter)
        random.seed(1)
        sbj_grp0 = set(random.sample(sbj_set, k=n_grp0))
        sbj_grp1 = sbj_set - sbj_grp0

        # build file_tree of each group
        file_tree_list = list()
        tqdm_dict = {'total': 2,
                     'desc': 'aggregate stats per subgroup',
                     'disable': not verbose}

        # todo: clumsy, maybe load after split?
        ijk_set = set(self.ijk_fs_dict.keys())
        if not ijk_set:
            ijk_set = None

        for sbj_grp in tqdm((sbj_grp0, sbj_grp1), **tqdm_dict):
            ft = self.get_subset(sbj_grp, ijk_set=ijk_set, **kwargs,
                                 verbose=verbose)
            file_tree_list.append(ft)

        return file_tree_list
