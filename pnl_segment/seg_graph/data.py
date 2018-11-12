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
        sbj_ijk_fs_dict (dict): tree, keys are sbj, ijk.  leafs are feat stat
                                stores stats per voxel and sbj ... can be
                                set to None for memory savings by neuter(), but
                                then obj is not capable of split()
    """

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
        ijk_set = set(mask).intersection(self.ijk_fs_dict.keys())
        fs = sum(self.ijk_fs_dict[ijk] for ijk in ijk_set)
        fs_normal = fs.to_normal()

        # resample
        for ijk in set(mask).intersection(self.ijk_fs_dict.keys()):
            # sample same number of values
            n = self.ijk_fs_dict[ijk].n
            x = np.atleast_2d(fs_normal.rvs(n)).T

            # build new feat_stat and store
            fs = FeatStat.from_array(x)
            self.ijk_fs_dict[ijk] = fs

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree=dict(), **kwargs):
        # init
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.ijk_fs_dict = defaultdict(FeatStatEmpty)
        self.sbj_ijk_fs_dict = defaultdict(dict)
        self.ref = None
        self.feat_list = None

        if sbj_feat_file_tree:
            self.load(**kwargs)

    def unload(self):
        self.sbj_ijk_fs_dict = None

    def load(self, **kwargs):
        if not self.sbj_feat_file_tree:
            raise AttributeError('sbj_feat_file_tree is empty, no data passed')
        self.add_nii(self.sbj_feat_file_tree, **kwargs)

    def apply_mask(self, mask):
        ft = FileTree()
        ft.ref = self.ref
        ft.feat_list = self.feat_list
        ft.sbj_feat_file_tree = self.sbj_feat_file_tree

        ijk_set = mask.to_point_cloud()
        ijk_set &= {x for x in self.ijk_fs_dict.keys()}
        for ijk in ijk_set:
            ft.ijk_fs_dict[ijk] = self.ijk_fs_dict[ijk]

            # copy sbj specific info
            for sbj in self.sbj_iter:
                if ijk in self.sbj_ijk_fs_dict[sbj].keys():
                    ft.sbj_ijk_fs_dict[sbj][ijk] = \
                        self.sbj_ijk_fs_dict[sbj][ijk]

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

    def add_nii(self, sbj_feat_file_tree, verbose=False):
        """ adds sbj to dataset

        Args:
            sbj_feat_file_tree (dict): dict of dict, key0: sbj, key1: feat,
                                       value: file
            verbose (bool): toggles command line output
        """
        # init self.ref if needed
        f_nii_dict = next(iter(sbj_feat_file_tree.values()))
        if self.ref is None:
            f_nii = next(iter(f_nii_dict.values()))
            self.ref = get_ref(f_nii)

        # init self.feat_list if needed
        if self.feat_list is None:
            self.feat_list = sorted(f_nii_dict.keys())

        # store files
        tqdm_dict = {'disable': not verbose,
                     'desc': 'load sbj'}
        for sbj, f_nii_dict in tqdm(sbj_feat_file_tree.items(), **tqdm_dict):
            # ensure appropriate features
            if set(f_nii_dict.keys()) != set(self.feat_list):
                raise AttributeError(f'feat must be {self.feat_list} in {sbj}')

            # store file
            self.sbj_feat_file_tree[sbj] = f_nii_dict

            # get data
            ijk_data_dict = self.get_ijk_data_dict(sbj_list=[sbj])

            # update stats per sbj
            for ijk, x in ijk_data_dict.items():
                # compute
                fs = FeatStat.from_array(x, obs_greater_dim=False)

                # store per sbj
                self.sbj_ijk_fs_dict[sbj][ijk] = fs

                # store aggregate stats
                self.ijk_fs_dict[ijk] += fs

    def get_ijk_data_dict(self, sbj_list=None, mask=None, verbose=False):
        """ returns data per ijk, loads from original files

        Args:
            sbj_list (list): list of sbj to include
            mask (Mask): restrict ijk values in return dict
            verbose (bool): toggles command line output

        Returns:
            ijk_data_dict (dict): keys are ijk tuple, values are array of data
                                  with shape (n, len(self.feat_list))
        """
        if sbj_list is None:
            sbj_list = list(self.sbj_iter)

        if mask is None:
            f_nii_list = list()
            for sbj in sbj_list:
                f_nii_list += list(self.sbj_feat_file_tree[sbj].values())
            mask = sum(Mask.from_nii(f_nii) for f_nii in f_nii_list)

        tqdm_dict = {'disable': not verbose,
                     'desc': 'load data per sbj'}
        ijk_list_dict = defaultdict(list)
        for sbj in tqdm(sbj_list, **tqdm_dict):
            # validate reference space
            f_nii_list = [self.sbj_feat_file_tree[sbj][feat]
                          for feat in self.feat_list]
            if any(get_ref(f_nii) != self.ref for f_nii in f_nii_list):
                raise AttributeError('reference space mismatch')

            # load data
            data = [nib.load(str(f)).get_data() for f in f_nii_list]
            data = np.stack(data, axis=3)

            # store data
            for ijk in mask.to_point_cloud():
                x = data[ijk[0], ijk[1], ijk[2], :]
                if x.all():
                    # only add vectors if each element is positive
                    ijk_list_dict[ijk].append(x)

        # build into array (this is comp expensive ... allocation)
        tqdm_dict = {'disable': not verbose,
                     'desc': 'allocate data array per ijk'}
        ijk_data_dict = dict()
        for ijk, l in tqdm(ijk_list_dict.items(), **tqdm_dict):
            ijk_data_dict[ijk] = np.stack(l, axis=0)

        return ijk_data_dict

    def get_mask(self, p=0):
        """ returns a mask which has at least p percent of sbj included
        """
        x = np.zeros(self.ref.shape).astype(bool)
        n = len(list(self.sbj_iter)) * p
        for ijk, fs in self.ijk_fs_dict.items():
            if fs.n >= n:
                x[ijk] = True
        return Mask(x, ref=self.ref)

    def get_point_cloud(self, p=0):
        """ returns a point_cloud which has at least p percent of sbj included
        """
        pc = PointCloud({}, ref=self.ref)
        n = len(list(self.sbj_iter)) * p
        for ijk, fs in self.ijk_fs_dict.items():
            if fs.n >= n:
                pc.add(ijk)
        return pc

    def split(self, p=.5, unload_self=False, unload_kids=True, verbose=False):
        """ splits data into two groups

        Args:
            p (float): in (0, 1), splits data to have at least p percent of sbj
                       in the first grp
            unload_self (bool): toggles if sbj level stats discarded in self
            unload_kids (bool): toggles if sbj level stats discarded in output
            verbose (bool): toggles command line output

        Returns:
            file_tree_list (list): FileTree for each grp
        """

        if self.sbj_ijk_fs_dict is None:
            raise AttributeError('per sbj data not present, call load() first')

        # compute n_effect
        n_grp0 = np.ceil(p * len(self)).astype(int)

        # split sbj into health and effect groups
        sbj_set = set(self.sbj_iter)
        sbj_grp0 = random.sample(sbj_set, k=n_grp0)
        sbj_grp1 = sbj_set - set(sbj_grp0)

        # build file_tree of each group
        file_tree_list = list()
        pbar = tqdm(total=len(sbj_set),
                    desc='aggregate stats per sbj',
                    disable=not verbose)
        for sbj_grp in (sbj_grp0, sbj_grp1):
            # init new file tree obj, copy relevant fields from self
            ft = FileTree()
            ft.ref = self.ref
            ft.feat_list = self.feat_list

            for sbj in sbj_grp:
                # save files
                ft.sbj_feat_file_tree[sbj] = self.sbj_feat_file_tree[sbj]

                # store ijk_fs_dict per sbj
                try:
                    ijk_fs_dict = self.sbj_ijk_fs_dict[sbj]
                except KeyError:
                    err_msg = 'keep_sbj=False in previous call to split()'
                    raise AttributeError(err_msg)

                # add contribution to summary stats
                for ijk, fs in ijk_fs_dict.items():
                    ft.ijk_fs_dict[ijk] += fs
                    if not unload_kids:
                        ft.sbj_ijk_fs_dict[sbj][ijk] = fs
                pbar.update(1)
            file_tree_list.append(ft)

        if unload_self:
            self.unload()

        return file_tree_list
