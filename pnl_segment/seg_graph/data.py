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

    @staticmethod
    def harmonize_via_add(ft_list, apply=True, verbose=False):
        """ ensures that each file tree has same mean (over all ijk)

        average is determined by summing across all file tree and ijk

        Args:
            ft_list (list): list of file trees
            apply (bool): toggles whether to apply offset
            verbose (bool): toggles command line output

        Returns:
            mu_offset_dict (dict): keys are file tree, values are offset needed
        """

        # find average per file tree
        ft_fs_dict = defaultdict(FeatStatEmpty)
        tqdm_dict0 = {'disable': not verbose,
                     'desc': 'compute fs sum per group',
                     'total': len(ft_list)}
        for ft in tqdm(ft_list, **tqdm_dict0):
            tqdm_dict1 = {'disable': not verbose,
                          'desc': 'summing fs per voxel',
                          'total': len(ft.ijk_fs_dict)}
            for fs in tqdm(ft.ijk_fs_dict.values(), **tqdm_dict1):
                ft_fs_dict[ft] += fs

        # find average (over all file trees)
        fs_average = sum(ft_fs_dict.values())

        # compute mu_offset per ft
        mu_offset_dict = {ft: fs_average.mu - fs.mu
                          for ft, fs in ft_fs_dict.items()}

        # apply (if need be)
        if apply:
            for ft, mu_delta in mu_offset_dict.items():
                for ijk in ft.ijk_fs_dict.keys():
                    ft.ijk_fs_dict[ijk].mu += mu_delta

        return mu_offset_dict

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.get_mask()

        return self._mask

    @property
    def sbj_iter(self):
        return iter(self.sbj_feat_file_tree.keys())

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree, mask=None):
        # todo: all attributes should be internal
        # init
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.ijk_fs_dict = dict()
        self._mask = mask
        if isinstance(self._mask, Mask):
            self._mask = PointCloud.from_mask(self._mask)

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

    def load(self, mask=None, verbose=False, **kwargs):
        """ loads files, adds data to statistics per voxel

        Args:
            mask (Mask): ijk tuples to load
            verbose (bool): toggles command line output
        """
        # get mask_ijk
        if mask is None:
            mask = self.get_mask(**kwargs)

        # load all data
        data = self.get_array(verbose=verbose)

        # compute feat stat
        tqdm_dict = {'disable': not verbose,
                     'desc': 'compute feat stat'}
        self.ijk_fs_dict = dict()

        for ijk in tqdm(PointCloud.from_mask(mask), **tqdm_dict):
            x = data[ijk[0], ijk[1], ijk[2], :, :]
            self.ijk_fs_dict[ijk] = FeatStat.from_array(x)

    def unload(self):
        self.ijk_fs_dict = dict()

    def __reduce_ex__(self, *args, **kwargs):
        # store and remove ijk_fs_dict from self
        ijk_fs_dict = self.ijk_fs_dict
        self.ijk_fs_dict = dict()

        # save without ijk_fs_dict in self
        x = super().__reduce_ex__(*args, **kwargs)

        # put it back
        self.ijk_fs_dict = ijk_fs_dict

        return x

    def apply_mask(self, mask):
        ft = FileTree(self.sbj_feat_file_tree)
        ft.ijk_fs_dict = defaultdict(FeatStatEmpty)

        ijk_set = PointCloud.from_mask(mask)
        ijk_set &= {x for x in self.ijk_fs_dict.keys()}
        for ijk in ijk_set:
            ft.ijk_fs_dict[ijk] = self.ijk_fs_dict[ijk]

        return ft

    def get_array(self, verbose=False):
        """ returns an array of all data

        Returns:
            data (np.array): shape=(space0, space1, space2, num_feat, num_sbj)
        """

        # preallocate
        num_feat = len(self.feat_list)
        num_sbj = len(self.sbj_feat_file_tree.items())
        f_any = next(iter(self.sbj_feat_file_tree.values()))[self.feat_list[0]]
        shape = nib.load(str(f_any)).shape
        data = np.empty((shape[0], shape[1], shape[2], num_feat, num_sbj))

        tqdm_dict = {'disable': not verbose,
                     'desc': 'load data',
                     'total': num_sbj}

        idx_sbj_dict_iter = enumerate(self.sbj_feat_file_tree.items())
        for sbj_idx, (sbj, f_nii_dict) in tqdm(idx_sbj_dict_iter, **tqdm_dict):
            for feat_idx, feat in enumerate(self.feat_list):
                img = nib.load(str(f_nii_dict[feat]))
                data[:, :, :, feat_idx, sbj_idx] = img.get_data()
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

            def fnc(reg):
                return reg.mu.flatten()[feat_idx]

        for ijk, reg in self.ijk_fs_dict.items():
            x[ijk] = fnc(reg)

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
        for sbj in self.sbj_iter:
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

        Returns:
            file_tree_list (list): FileTree for each grp
        """

        # compute n_effect
        n_grp0 = np.ceil(p * len(self)).astype(int)

        # split sbj into health and effect groups
        sbj_set = set(self.sbj_iter)
        if seed is not None:
            random.seed(seed)
        sbj_grp0 = set(random.sample(sbj_set, k=n_grp0))
        sbj_grp1 = sbj_set - sbj_grp0

        # build file_tree of each group
        file_tree_list = list()
        for sbj_grp in (sbj_grp0, sbj_grp1):
            sbj_feat_file_tree = {sbj: self.sbj_feat_file_tree[sbj]
                                  for sbj in sbj_grp}
            ft = FileTree(sbj_feat_file_tree)
            file_tree_list.append(ft)

        return file_tree_list
