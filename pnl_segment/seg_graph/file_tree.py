import random
from collections import defaultdict

import nibabel as nib
import numpy as np
from tqdm import tqdm

from ..region.feat_stat import FeatStatEmpty, FeatStat
from ..space import get_ref, Mask, PointCloud


class FileTree:
    """ manages scalar files of 3d volumes

    the emphasis is on storing feature statistics per voxel across the
    population (see ijk_fs_dict)

    Attributes:
        sbj_feat_file_tree (tree): tree, keys are sbj, feature, leafs are files
        feat_list (list): list of features, defines ordering of feat
        ref (RefSpace): defines shape and affine of data
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
                # apply to feature stats
                for ijk in pc:
                    ft.ijk_fs_dict[ijk].mu += mu_delta

                    i, j, k = ijk
                    ft.data[i, j, k, :, :] += mu_delta

        return mu_offset_dict

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree, mask=None):
        # todo: all attributes should be internal
        # init
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())

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

    def load(self, verbose=False, **kwargs):
        """ loads files, adds data to statistics per voxel

        Args:
            verbose (bool): toggles command line output
        """

        # load all data
        self.data = self.get_data(verbose=verbose, mask=self.mask)

        # compute feat stat
        tqdm_dict = {'disable': not verbose,
                     'desc': 'compute feat stat'}
        self.ijk_fs_dict = dict()
        for ijk in tqdm(PointCloud.from_mask(self.mask), **tqdm_dict):
            x = self.data[ijk[0], ijk[1], ijk[2], :, :].T
            self.ijk_fs_dict[ijk] = FeatStat.from_array(x)

    def unload(self):
        self.ijk_fs_dict = dict()
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
