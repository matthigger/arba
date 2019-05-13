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
        data (np.memmap): if data is loaded, a read only memmap of data
        split_effect (tuple): split, effect

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

    def __init__(self, sbj_feat_file_tree, fnc_list=None, mask=None):
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())
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

        self.split_effect = None

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
    def get_fs(self, ijk, sbj_list=None, sbj_bool=None):

        # get sbj_bool
        assert not ((sbj_list is not None) and (sbj_bool is not None)), \
            'nand(sbj_list, sbj_bool) required'
        if sbj_bool is None:
            sbj_bool = self.sbj_list_to_bool(sbj_list)

        # build fs
        i, j, k = ijk
        x = self.data[i, j, k, :, :]

        # apply effect
        if self.split_effect is not None:
            split, effect = self.split_effect
            if effect.mask[i, j, k]:
                x = np.array(x)
                x[split, :] += effect.mean

        x = x[sbj_bool, :].reshape((-1, self.d), order='F')
        return FeatStat.from_array(x.T)

    @contextmanager
    def loaded(self, **kwargs):
        """ provides context manager with loaded data

        note: we only load() and unload() if the object was previously not
        loaded.  otherwise we're all set, no need to do it again.
        """
        was_loaded = self.is_loaded
        if not was_loaded:
            self.load(**kwargs)
        try:
            yield self
        finally:
            if not was_loaded:
                self.unload()

    def load(self, verbose=False):
        """ loads data, applies fnc in self.fnc_list
        """
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

        # flush data to memmap, make read only copy
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
        os.remove(str(self.f_data))
        self.data = None
        self.f_data = None
        self.__mask = None

    @check_loaded
    def get_data_subset(self, sbj_list=None):
        """ returns view of data corresponding only to sbj in sbj_list
        """
        raise NotImplementedError

    @check_loaded
    def to_nii(self, fnc=None, feat=None, sbj_list=None, f_out=None,
               sbj_bool=None, back=0):
        """ writes data to a nii file

        Args:
            fnc (fnc): accepts data array, returns a 3d volume
            feat : some feature in feat_list, if passed, fnc returns mean feat
            sbj_list (list): list of sbj to include, if None all sbj used
            f_out (str or Path): output nii file
            back (float): background value

        Returns:
            f_out (Path): nii file out
        """
        # get f_out
        if f_out is None:
            f_out = tempfile.NamedTemporaryFile(suffix='.nii.gz').name

        # get sbj_bool
        assert (sbj_list is None) != (sbj_bool is None), \
            'sbj_list xor sbj_bool'
        if sbj_bool is None:
            assert set(self.sbj_list).issuperset(sbj_list), \
                'sbj not in FileTree'
            sbj_bool = self.sbj_list_to_bool(sbj_list)

        # build img
        if fnc is not None:
            # build based on custom fnc
            raise NotImplementedError('how does fnc handle effect?')
            x = back * np.ones(self.ref.shape)
            for i, j, k in self.mask.iter_ijk():
                x[i, j, k] = fnc(self.data[i, j, k, sbj_bool, :])
        else:
            # build based on mean feature
            feat_idx = self.feat_list.index(feat)
            x = self.data[:, :, :, sbj_bool, feat_idx]
            if self.split_effect is not None:
                x = np.array(x)
                split, effect = self.split_effect
                # todo: all splits are numpy array
                # todo: rename sbj_bool -> split
                # index effect into only active sbj
                _split = np.array(split)[np.array(sbj_bool)]
                if _split.sum():
                    _x = x[effect.mask, :]
                    _x[:, _split] += effect.mean[feat_idx]
                    x[effect.mask, :] = _x
            x = np.mean(x, axis=3)

        # write to file
        img = nib.Nifti1Image(x, affine=self.ref.affine)
        img.to_filename(str(f_out))

        return pathlib.Path(f_out)

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
