import os
import pathlib
import tempfile
from contextlib import contextmanager

import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba.region import FeatStat
from arba.space import get_ref, Mask


def check_loaded(fnc):
    def wrapped(self, *args, **kwargs):
        assert self.is_loaded, 'FileTree must be loaded to call'
        return fnc(self, *args, **kwargs)

    return wrapped


class FileTree:
    """ manages large datasets of multivariate images

    the focus is on a context manager which loads data.  data is loaded into a
    data cube, operated on by some fncs, then memory mapped.  FileTree.data
    is then replaced with a read-only version of the memory mapped array,
    allowing for parallel processes to operate on shared memory.

    Attributes:
        sbj_feat_file_tree (tree): key0 are sbj, key2 are feat, values are file
        sbj_list (list): list of sbj (defines indexing)
        feat_list (list): list of features (defines indexing)
        scale (np.array): defines scaling of data
        ref (RefSpace): defines shape and affine of data
        fnc_list (list): each is a function which is passed self, may
                         operate on data as needed before it is write
                         protected

    Attributes available after load()
        mask (Mask): mask of active area
        fs (FeatStat): feature statistics across active area of all sbj
        data (np.memmap): (space0, space1, space2, sbj_idx, feat_idx)
                          if data is loaded, a read only memmap of data
        effect_list (list): list of effects applied

    Hidden Attributes:
        __mask (Mask): mask of intersection of all files (only available when
                       loaded)
    """

    @property
    def is_loaded(self):
        return self.data is not None

    @property
    def num_sbj(self):
        return len(self.sbj_feat_file_tree)

    @property
    def d(self):
        return len(self.feat_list)

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree, sbj_list=None, fnc_list=None,
                 mask=None):
        self.sbj_feat_file_tree = sbj_feat_file_tree
        if sbj_list is None:
            self.sbj_list = sorted(self.sbj_feat_file_tree.keys())
        else:
            self.sbj_list = sbj_list
            assert set(sbj_list) == set(sbj_feat_file_tree), 'sbj_list error'

        feat_file_dict = next(iter(self.sbj_feat_file_tree.values()))
        self.feat_list = sorted(feat_file_dict.keys())
        self.scale = np.eye(self.d)
        self.ref = get_ref(next(iter(feat_file_dict.values())))

        self.fnc_list = fnc_list
        if fnc_list is None:
            self.fnc_list = list()

        self.fs = None
        self.data = None
        self.f_data = None

        self.mask = mask
        self.__mask = None

        self.effect_list = list()

    def discard_to(self, n_sbj, split=None):
        """ discards sbj so only n_sbj remains (in place)

        Args:
            n_sbj (int):
            split (Split): if passed, ensures at most n_sbj per split grp
        """
        if self.is_loaded:
            raise AttributeError('may not be loaded during discard_to()')

        # get appropriate grp_list_iter, iter of grp, sbj_list
        if split is None:
            grp_list_iter = iter(('grp0', self.sbj_list))
        else:
            grp_list_iter = split.grp_list_iter()

        # prune tree to approrpiate sbj
        for _, sbj_list in grp_list_iter:
            for sbj in sbj_list[n_sbj:]:
                del self.sbj_feat_file_tree[sbj]

        # update sbj_list
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())

    @check_loaded
    def get_fs(self, ijk=None, mask=None, pc_ijk=None, sbj_list=None,
               sbj_bool=None):
        assert not ((sbj_list is not None) and (sbj_bool is not None)), \
            'nand(sbj_list, sbj_bool) required'
        assert 2 == ((ijk is None) + (mask is None) + (pc_ijk is None)), \
            'xor(ijk, mask, pc_ijk) required'

        # get sbj_bool
        if sbj_bool is None:
            if sbj_list is None:
                sbj_list = self.sbj_list
            sbj_bool = self.sbj_list_to_bool(sbj_list)

        # get data array
        if ijk is not None:
            # single point
            i, j, k = ijk
            x = self.data[i, j, k, :, :]
            x = x[sbj_bool, :].reshape((-1, self.d), order='F')
        elif mask is not None:
            # mask
            x = self.data[mask, :, :]
            x = x[:, sbj_bool, :].reshape((-1, self.d), order='F')
        else:
            n = len(pc_ijk)
            x = np.empty((n, self.d))
            for idx, (i, j, k) in enumerate(pc_ijk):
                _x = self.data[i, j, k, :, :]
                x[idx, :] = _x[sbj_bool, :].reshape((-1, self.d), order='F')

        return FeatStat.from_array(x.T)

    @contextmanager
    def loaded(self, effect_list=None, **kwargs):
        """ provides context manager with loaded data

        note: we only load() and unload() if the object was previously not
        loaded
        """
        was_loaded = self.is_loaded

        # load or check that previous load was equivilent
        if was_loaded:
            if effect_list is not None:
                assert effect_list == self.effect_list, \
                    'effect_list mismatch between load()'
        else:
            self.load(effect_list=effect_list, **kwargs)

        try:
            yield self
        finally:
            if not was_loaded:
                self.unload()

    def load(self, effect_list=None, verbose=False, memmap=False):
        """ loads data, applies fnc in self.fnc_list

        Args:
            effect_list = (list): list of effects
            verbose (bool): toggles command line output
            memmap (bool): toggles memory map (self.data becomes read only, but
                           accesible from all threads to save on memory)
        """

        if self.is_loaded:
            raise AttributeError('already loaded')

        # build memmap file
        shape = (*self.ref.shape, self.num_sbj, self.d)
        self.data = np.zeros(shape)

        # load data
        for sbj_idx, sbj in tqdm(enumerate(self.sbj_list),
                                 desc='load per sbj',
                                 disable=not verbose):
            for feat_idx, feat in enumerate(self.feat_list):
                f = self.sbj_feat_file_tree[sbj][feat]
                img = nib.load(str(f))
                self.data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # get mask of data
        self.__mask = Mask(np.all(self.data, axis=(3, 4)), ref=self.ref)

        # get mask
        if self.mask is not None:
            self.mask = np.logical_and(self.__mask, self.mask)
        else:
            self.mask = self.__mask

        # apply all fnc
        for fnc in self.fnc_list:
            fnc(self)

        # apply effects
        if effect_list is None:
            self.effect_list = list()
        else:
            self.effect_list = effect_list
        for effect in self.effect_list:
            self.data += effect.get_offset_array(sbj_list=self.sbj_list)

        # flush data to memmap, make read only copy
        if memmap:
            self.f_data = tempfile.NamedTemporaryFile(suffix='.dat').name
            self.f_data = pathlib.Path(self.f_data)
            x = np.memmap(self.f_data, dtype='float32', mode='w+',
                          shape=shape)
            x[:] = self.data[:]
            x.flush()
            self.data = np.memmap(self.f_data, dtype='float32', mode='r',
                                  shape=shape)

    def unload(self):
        # delete memory map file
        if self.f_data is not None and self.f_data.exists():
            os.remove(str(self.f_data))
            self.f_data = None

        self.data = None
        self.__mask = None
        self.effect_list = list()

    @check_loaded
    def to_nii(self, folder=None):
        """ writes each feature to a nii file

        Args:
            folder (str or Path): output folder

        Returns:
            f_out (Path): nii file out
        """
        if folder is None:
            folder = tempfile.TemporaryDirectory().name
        folder = pathlib.Path(folder)

        # write to file
        for feat_idx, feat in enumerate(self.feat_list):
            x = self.data[:, :, :, :, feat_idx]
            img = nib.Nifti1Image(x, affine=self.ref.affine)
            img.to_filename(str(folder / f'{feat}.nii.gz'))

        return folder

    def sbj_list_to_bool(self, sbj_list=None):
        if sbj_list is None:
            return np.ones(self.num_sbj).astype(bool)

        sbj_set = set(sbj_list)
        return np.array([sbj in sbj_set for sbj in self.sbj_list])

    def __eq__(self, other):
        if self.sbj_feat_file_tree != other.sbj_feat_file_tree:
            return False

        if self.ref != other.ref:
            return False

        if self.mask != other.mask:
            return False

        if self.scale != other.scale:
            return False

        if self.fnc_list != other.fnc_list:
            return False

        return True


def scale_normalize(ft):
    """ compute & store mean and var per feat, scale + offset to Z score

    Args:
        ft (FileTree): file tree to be equalized
    """
    # compute stats
    shape = (len(ft.mask) * ft.num_sbj, ft.d)
    shape_orig = (len(ft.mask), ft.num_sbj, ft.d)
    _data = ft.data[ft.mask, :, :].reshape(shape, order='F')
    ft.fs = FeatStat.from_array(_data.T)

    # apply scale equalization
    scale = np.diag(1 / np.diag(ft.fs.cov)) ** .5
    _data = (_data - ft.fs.mu) @ scale
    ft.data[ft.mask, :, :] = _data.reshape(shape_orig, order='F')
