import multiprocessing
import random
from copy import deepcopy

import nibabel as nib
import numpy as np
from tqdm import tqdm

from mh_pytools.parallel import run_par_fnc
from ..region.feat_stat import FeatStatEmpty, FeatStat
from ..space import get_ref, Mask, PointCloud


def get_ijk_fs_dict(ijk_list, x, n_cpu=None, verbose=False, par_flag=False):
    if par_flag is False:
        return _get_ijk_fs_dict(ijk_list=ijk_list, x=x, verbose=verbose)

    if n_cpu is None:
        n_cpu = multiprocessing.cpu_count()

    # split into sub-matrices along first axis
    ijk_list = sorted(ijk_list)
    idx_list = np.linspace(0, len(ijk_list), n_cpu + 1).astype(int)

    # build arg_list to get_ijk_fs_dict
    arg_list = list()
    for idx0, idx1 in zip(idx_list, idx_list[1:]):
        _ijk_list = ijk_list[idx0: idx1 + 1]
        offset = np.array((_ijk_list[0][0], 0, 0))
        _x = x[_ijk_list[0][0]: _ijk_list[-1][0] + 1, ...]
        arg_list.append({'offset': offset,
                         'ijk_list': _ijk_list,
                         'x': _x,
                         'verbose': False})

    # run
    res_out = run_par_fnc(_get_ijk_fs_dict, arg_list,
                          verbose=verbose)

    # aggregate par output
    ijk_fs_dict = res_out[0]
    for _ijk_fs_dict in res_out[1:]:
        ijk_fs_dict.update(_ijk_fs_dict)

    return ijk_fs_dict


def _get_ijk_fs_dict(ijk_list, x, verbose=False, offset=None):
    if offset is None:
        offset = np.array((0, 0, 0))

    tqdm_dict = {'disable': not verbose,
                 'desc': 'compute feat stat'}
    ijk_fs_dict = dict()
    for ijk in tqdm(ijk_list, **tqdm_dict):
        _ijk = np.array(ijk) - offset
        _x = x[_ijk[0], _ijk[1], _ijk[2], :, :].T
        ijk_fs_dict[ijk] = FeatStat.from_array(_x)
    return ijk_fs_dict


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

        scale (np.array): square np.array, sets scaling of data via
                          scale_data() and scale_equalize()
    """

    def __or__(self, other):
        """ aggregates file tree

        NOTE: history not preserved

        Args:
            other (FileTree):

        Returns:
            ft (FileTree):
        """
        sbj_feat_file_tree = deepcopy(self.sbj_feat_file_tree)
        sbj_feat_file_tree.update(other.sbj_feat_file_tree)

        assert np.allclose(self.mask, other.mask), 'masks must match to add'
        assert self.ref == other.ref, 'refs must match to add'
        assert self.feat_list == other.feat_list, 'feat_list must match to add'

        ft = FileTree(sbj_feat_file_tree, mask=self.mask, ref=self.ref,
                      feat_list=deepcopy(self.feat_list))

        # sum feat stats
        if self.ijk_fs_dict and other.ijk_fs_dict:
            for ijk in self.ijk_fs_dict.keys():
                ft.ijk_fs_dict[ijk] = self.ijk_fs_dict[ijk] + \
                                      other.ijk_fs_dict[ijk]

        # sum data (if available)
        if self.data is None or other.data is None:
            ft.data is None
        else:
            # add in data
            shape = list(self.data.shape)
            shape[3] = len(ft.sbj_list)
            ft.data = np.empty(shape)

            for _ft in (self, other):
                bool_idx = [sbj in _ft.sbj_list for sbj in ft.sbj_list]
                ft.data[:, :, :, bool_idx, :] = _ft.data

        return ft

    def sum_feat_stat(self, mask=None, verbose=False):
        if mask is None:
            mask = self.mask
        pc = PointCloud.from_mask(mask)

        # find average per file tree
        fs = FeatStatEmpty()
        tqdm_dict1 = {'disable': not verbose,
                      'desc': 'summing fs per voxel',
                      'total': len(self.ijk_fs_dict)}
        for ijk in tqdm(pc, **tqdm_dict1):
            fs += self.ijk_fs_dict[ijk]

        return fs

    @staticmethod
    def scale_equalize(ft_tuple, verbose=False, mask=None):
        """ ensures that each file tree has same mean (over all ijk)

        average is determined by summing across all file tree and ijk

        Args:
            ft_tuple (tuple): file trees, length 2
            verbose (bool): toggles command line output
            mask (np.array): where to harmonize

        Returns:
            mu_offset_dict (dict): keys are file tree, values are offset needed
        """
        # compute average feature
        ft_fs_dict = {ft: ft.sum_feat_stat(mask, verbose) for ft in ft_tuple}
        fs_average = sum(ft_fs_dict.values())

        # compute scale
        scale = np.diag(np.diag(fs_average.cov) ** (-.5))
        if np.isnan(scale).any():
            raise RuntimeError('invalid scale')
        for ft in ft_tuple:
            ft.scale_data(scale)

    def scale_data(self, scale=None):
        if scale is None:
            # unscale
            if self.scale is None:
                # no scale set, nothing to unscale
                return
            else:
                scale = np.linalg.inv(scale)

        # record scale
        if self.scale is None:
            self.scale = scale
        else:
            self.scale = scale @ self.scale

        for ijk, fs in self.ijk_fs_dict.items():
            self.ijk_fs_dict[ijk] = fs.scale(scale)

    @staticmethod
    def harmonize_via_add(ft_tuple, apply=True, verbose=False, mask=None):
        """ ensures that each file tree has same mean (over all ijk)

        average is determined by summing across all file tree and ijk

        Args:
            ft_tuple (tuple): file trees, length 2
            apply (bool): toggles whether to apply offset
            verbose (bool): toggles command line output
            mask (np.array): where to harmonize

        Returns:
            mu_offset_dict (dict): keys are file tree, values are offset needed
        """
        ft0, ft1 = ft_tuple
        if mask is None:
            assert np.allclose(ft0.mask,
                               ft1.mask), 'mask mismatch'
            mask = ft0.mask

        # average features, compute average
        ft_fs_dict = {ft: ft.sum_feat_stat(mask, verbose) for ft in ft_tuple}
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

    def __eq__(self, other):
        if self.sbj_feat_file_tree != other.sbj_feat_file_tree:
            return False

        if self.ref != other.ref:
            return False

        if self.mask != other.mask:
            return False

        if self.add_hist_list != other.add_hist_list:
            return False

        if self.scale != other.scale:
            return False

        return True

    def __init__(self, sbj_feat_file_tree, mask=None, ref=None,
                 feat_list=None):
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
        self.feat_list = feat_list
        if self.feat_list is None:
            for sbj in self.sbj_list:
                feat_list = sorted(self.sbj_feat_file_tree[sbj].keys())
                if self.feat_list is None:
                    self.feat_list = feat_list
                elif self.feat_list != feat_list:
                    raise AttributeError('feat_list mismatch')

        # init null scale
        self.scale = np.eye(len(self.feat_list))

        # assign and validate ref space
        self.ref = ref
        if self.ref is None:
            for f_nii_dict in self.sbj_feat_file_tree.values():
                for f_nii in f_nii_dict.values():
                    if self.ref is None:
                        self.ref = get_ref(f_nii)
                    elif self.ref != get_ref(f_nii):
                        raise AttributeError('space mismatch')

    def reset_hist(self):
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

    def load(self, verbose=False, load_data=True, load_ijk_fs=True, _data=None,
             par_flag=False, **kwargs):
        """ loads files, adds data to statistics per voxel

        Args:
            load_data (bool): toggles whether data is loaded (very heavy)
            load_ijk_fs (bool): toggles wehther feature stats per voxel are
                                loaded (heavy)
            _data (np.ndarray): if passed, used in place of self.get_data()'s
                                return value.  useful internally in split()
            verbose (bool): toggles command line output
            par_flag (bool): toggles parallel loading
        """

        # load all data
        if _data is None:
            self.data = self.load_data(verbose=verbose, mask=self.mask)
        else:
            self.data = _data

        # compute feat stat
        if load_ijk_fs:
            pc = PointCloud.from_mask(self.mask)
            self.ijk_fs_dict = get_ijk_fs_dict(ijk_list=pc,
                                               x=self.data,
                                               verbose=verbose,
                                               par_flag=par_flag)

        # add values from history
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

    def load_data(self, verbose=False, mask=None):
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

            unscale = np.linalg.inv(self.scale)

            def fnc(fs):
                return (unscale @ fs.mu)[feat_idx]

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
        mask = np.stack((Mask.from_nii(f_nii) for f_nii in f_nii_list))
        mask = np.mean(mask.astype(bool), axis=0) >= p

        return Mask(mask, ref=get_ref(f_nii_list[0]))

    def split(self, p=None, n=None, sbj_set0=None, seed=None, par_flag=False,
              verbose=False):
        """ splits data into two groups

        Args:
            p (float): in (0, 1), splits data to have at least p percent of sbj
                       in the first grp
            n (int): number of sbj in the first file_tree returned
            sbj_set0 (set): set of sbj in first file_tree returned
            seed: sets random seed
            par_flag (bool): toggles parallel
            verbose (bool): toggles cmd line output

        Returns:
            file_tree_list (list): FileTree for each grp
        """
        if (p is None) == (n is None) == (sbj_set0 is None):
            raise AttributeError('either p xor n xor sbj_set required')

        # reset seed
        if seed is not None:
            random.seed(seed)

        # get sbj_set0
        sbj_set = set(self.sbj_list)
        if sbj_set0 is None:
            if n is None:
                n = np.ceil(p * len(self)).astype(int)

            # split sbj into health and effect groups
            sbj_set0 = set(random.sample(sbj_set, k=n))

        sbj_set1 = sbj_set - set(sbj_set0)

        # build file_tree of each group
        file_tree_list = list()
        for sbj_grp in (sbj_set0, sbj_set1):
            sbj_feat_file_tree = {sbj: self.sbj_feat_file_tree[sbj]
                                  for sbj in sbj_grp}
            ft = FileTree(sbj_feat_file_tree, mask=self.mask)
            file_tree_list.append(ft)

            # build ft.ijk_fs_dict if self has data
            if self.data is not None:
                # get relevant sbj_idx
                sbj_idx = [idx for idx, sbj in enumerate(self.sbj_list)
                           if sbj in ft.sbj_list]

                pc = PointCloud.from_mask(self.mask)
                x = self.data[:, :, :, sbj_idx, :]
                ft.ijk_fs_dict = get_ijk_fs_dict(ijk_list=pc,
                                                 x=x,
                                                 verbose=verbose,
                                                 par_flag=par_flag)

        return file_tree_list
