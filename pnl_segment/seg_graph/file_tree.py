import random
from collections import defaultdict
from copy import deepcopy

import nibabel as nib
import numpy as np
from tqdm import tqdm

from ..region.feat_stat import FeatStatEmpty, FeatStat
from ..space import get_ref, Mask, PointCloud


class FileTree:
    """ manages scalar files of 3d volumes

    minimizing memory usage is prioritized.  the last two attributes in the
    list below are memory intensive.  load() and unload() can be used to
    discard them when they are not needed.  add_hist_list serves as a memory of
    all the addition operations which have occurred to the data, such that
    calling load() will redo each addition operation in the history to ensure
    the state is not lost.

    Attributes:
        sbj_feat_file_tree (tree): tree, keys are sbj, feature, leafs are files
        sbj_list (list): list of all subjects, redundant, but other
                         applications may need this order fixed (dict is fishy
                         for this purpose)
        mask (Mask): a mask of the active area, this is fixed at __init__
        feat_list (list): list of features, defines ordering of feat
        ref (RefSpace): defines shape and affine of data

        add_hist_list (list): a list of tuples (value, mask) which are memory
                              of all additions which have been applied.  value
                              is np.array of length feat_list, mask is np.array
                              of booleans which describe location of addition

        ijk_fs_dict (dict): keys are ijk tuple, values are FeatStat objs
        data (np.array): shape is (space0, space1, space2, num_sbj, num_feat)
    """

    @staticmethod
    def harmonize_via_add(ft_list, apply=True, verbose=False, mask=None):
        """ ensures that each file tree has same mean (over all ijk)

        average is determined by summing across all file tree and ijk

        Args:
            ft_list (list): list of file trees
            apply (bool): toggles whether to apply offset
            verbose (bool): toggles command line output
            mask (np.array): where to harmonize

        Returns:
            mu_offset_dict (dict): keys are file tree, values are offset needed
        """
        if mask is None:
            assert np.allclose(ft_list[0].mask, ft_list[1].mask), \
                'mask mismatch'
            mask = ft_list[0].mask
        pc = PointCloud.from_mask(mask)

        # find average per file tree
        ft_fs_dict = defaultdict(FeatStatEmpty)
        tqdm_dict0 = {'disable': not verbose,
                      'desc': 'compute fs sum per group',
                      'total': len(ft_list)}
        for ft in tqdm(ft_list, **tqdm_dict0):
            tqdm_dict1 = {'disable': not verbose,
                          'desc': 'summing fs per voxel',
                          'total': len(ft.ijk_fs_dict)}
            for ijk in tqdm(pc, **tqdm_dict1):
                ft_fs_dict[ft] += ft.ijk_fs_dict[ijk]

        # find average (over all file trees)
        fs_average = sum(ft_fs_dict.values())

        # compute mu_offset per ft
        mu_offset_dict = {ft: fs_average.mu - fs.mu
                          for ft, fs in ft_fs_dict.items()}

        # apply (if need be)
        if apply:
            for ft, mu_delta in mu_offset_dict.items():
                ft.add(mu_delta, mask=mask)

        return mu_offset_dict

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree, mask=None):
        # init
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())
        self.add_hist_list = list()

        self.mask = mask
        if mask is None:
            self.mask = self.get_mask(p=1)

        self.ijk_fs_dict = dict()
        self.data = None

        # assign and validate feat_list
        self.feat_list = None
        for sbj in self.sbj_list:
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

    def reset(self):
        self.add_hist_list = list()

    def add(self, value, point_cloud=None, mask=None, record=True):
        # get point_cloud and mask
        if (point_cloud is None) == (mask is None):
            raise AttributeError('either point_cloud xor mask required')
        if point_cloud is not None:
            # point cloud passed, get mask
            mask = point_cloud.to_mask(self.ref)
        else:
            # mask passed, get point_cloud
            if isinstance(mask, np.ndarray):
                assert mask.shape == self.ref.shape, 'invalid shape'
                mask = Mask(mask, ref=self.ref)
            point_cloud = PointCloud.from_mask(mask)
        assert mask.ref == self.ref, 'invalid space'

        # store this addition (mask stored as its lighter for large volumes)
        if record:
            mask = deepcopy(np.array(mask))
            self.add_hist_list.append((value, mask))

        # add to feature statistics (if present)
        if self.ijk_fs_dict:
            for ijk in point_cloud:
                self.ijk_fs_dict[ijk].mu += value

        # add to data (if present)
        if self.data is not None:
            for i, j, k in point_cloud:
                self.data[i, j, k, :, :] += value

    def load(self, verbose=False, load_data=True, load_ijk_fs=True, **kwargs):
        """ loads files, adds data to statistics per voxel

        Args:
            load_data (bool): toggles whether data is loaded (very heavy)
            load_ijk_fs (bool): toggles wehther feature stats per voxel are
                                loaded (heavy)
            verbose (bool): toggles command line output
        """

        # load all data
        self.data = self.get_data(verbose=verbose, mask=self.mask)

        # compute feat stat
        if load_ijk_fs:
            tqdm_dict = {'disable': not verbose,
                         'desc': 'compute feat stat'}
            self.ijk_fs_dict = dict()
            for ijk in tqdm(PointCloud.from_mask(self.mask), **tqdm_dict):
                x = self.data[ijk[0], ijk[1], ijk[2], :, :].T
                self.ijk_fs_dict[ijk] = FeatStat.from_array(x)

        # add values into history
        for value, mask in self.add_hist_list:
            self.add(value=value, mask=mask, record=False)

        if not load_data:
            self.data = None

    def unload(self, unload_data=True, unload_ijk_fs=True):
        if unload_ijk_fs:
            self.ijk_fs_dict = dict()
        if unload_data:
            self.data = None

    def __reduce_ex__(self, *args, **kwargs):
        # store and remove ijk_fs_dict from self
        data = self.data
        ijk_fs_dict = self.ijk_fs_dict

        self.unload()

        # save without ijk_fs_dict in self
        x = super().__reduce_ex__(*args, **kwargs)

        # put it back
        self.ijk_fs_dict = ijk_fs_dict
        self.data = data

        return x

    def get_data(self, verbose=False, mask=None):
        """ returns an array of all data

        Returns:
            data (np.array): shape=(space0, space1, space2, num_sbj, num_feat)
        """

        # preallocate
        num_feat = len(self.feat_list)
        num_sbj = len(self.sbj_feat_file_tree.items())
        shape = self.ref.shape
        data = np.empty((shape[0], shape[1], shape[2], num_sbj, num_feat))

        tqdm_dict = {'disable': not verbose,
                     'desc': 'load data',
                     'total': num_sbj}

        # load files
        idx_sbj_dict_iter = enumerate(self.sbj_feat_file_tree.items())
        for sbj_idx, (sbj, f_nii_dict) in tqdm(idx_sbj_dict_iter, **tqdm_dict):
            for feat_idx, feat in enumerate(self.feat_list):
                img = nib.load(str(f_nii_dict[feat]))
                data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # apply mask
        if mask is not None:
            _mask = np.broadcast_to(mask.T, data.T.shape).T
            data[np.logical_not(_mask)] = 0

        return data

    def get_mean_array(self, fnc=None, feat=None):
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

            def fnc(fs):
                return fs.mu[feat_idx]

        for ijk, fs in self.ijk_fs_dict.items():
            x[ijk] = fnc(fs)

        return x

    def to_nii(self, f_out, **kwargs):
        x = self.get_mean_array(**kwargs)
        img = nib.Nifti1Image(x, self.ref.affine)
        img.to_filename(str(f_out))

    def get_mask(self, p=1):
        """ returns a mask which has all voxels with complete data
        """
        # require at least 1 observation per ijk
        p = max(p, np.nextafter(0, 1))

        # get set of all voxels present in all data file
        f_nii_list = list()
        for sbj in self.sbj_list:
            f_nii_list += list(self.sbj_feat_file_tree[sbj].values())

        # build mask
        mask = np.stack(Mask.from_nii(f_nii) for f_nii in f_nii_list)
        mask = np.mean(mask.astype(bool), axis=0) >= p

        return Mask(mask, ref=get_ref(f_nii_list[0]))

    def get_subset(self, sbj_list):
        # init new file tree obj from only sbj in sbj_list
        sbj_feat_file_tree = {sbj: self.sbj_feat_file_tree[sbj]
                              for sbj in sbj_list}
        ft = FileTree(sbj_feat_file_tree)
        return ft

    def split(self, p=.5, seed=None):
        """ splits data into two groups

        Args:
            p (float): in (0, 1), splits data to have at least p percent of sbj
                       in the first grp
            seed: sets random seed

        Returns:
            file_tree_list (list): FileTree for each grp
        """

        # compute n_effect
        n_grp0 = np.ceil(p * len(self)).astype(int)

        # reset seed
        if seed is not None:
            random.seed(seed)

        # split sbj into health and effect groups
        sbj_set = set(self.sbj_list)
        sbj_grp0 = set(random.sample(sbj_set, k=n_grp0))
        sbj_grp1 = sbj_set - sbj_grp0

        # build file_tree of each group
        file_tree_list = list()
        for sbj_grp in (sbj_grp0, sbj_grp1):
            sbj_feat_file_tree = {sbj: self.sbj_feat_file_tree[sbj]
                                  for sbj in sbj_grp}
            ft = FileTree(sbj_feat_file_tree, mask=self.mask)
            file_tree_list.append(ft)

        return file_tree_list
