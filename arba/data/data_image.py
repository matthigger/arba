import os
import pathlib
import tempfile
from contextlib import contextmanager

import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba.region import FeatStat
from arba.space import get_ref, Mask


class DataImage:
    """ manages large datasets of multivariate images

    the focus is on a context manager which loads data.  data is loaded into a
    data cube, operated on by some fncs, then memory mapped.  DataImage.data
    is then replaced with a read-only version of the memory mapped array,
    allowing for parallel processes to operate on shared memory.

    Attributes:
        sbj_ifeat_data_img (tree): key0 is sbj, key2 is imaging feat, vals are
                                    files which contain imaging feat of sbj
        sbj_list (list): list of sbj (defines indexing)
        feat_list (list): list of features (defines indexing)
        scale (np.array): defines scaling of data
        ref (RefSpace): defines shape and affine of data

    Attributes available after load()
        mask (Mask): mask of active area
        fs (FeatStat): feature statistics across active area of all sbj
        data (np.memmap): (space0, space1, space2, sbj_idx, feat_idx)
                          if data is loaded, a read only memmap of data
        offset (np.array): an offset which has been added to data

    Hidden Attributes:
        __mask (Mask): mask of intersection of all files (only available when
                       loaded)
    """

    @property
    def memmap(self):
        return bool(self.f_data)

    @property
    def is_loaded(self):
        return self.data is not None

    @property
    def num_sbj(self):
        return len(self.sbj_list)

    @property
    def d(self):
        return len(self.feat_list)

    def __len__(self):
        return len(self.sbj_list)

    def __init__(self, sbj_ifeat_data_img, sbj_list=None, mask=None,
                 memmap=False):
        self.sbj_ifeat_data_img = sbj_ifeat_data_img
        if sbj_list is None:
            self.sbj_list = sorted(self.sbj_ifeat_data_img.keys())
        else:
            self.sbj_list = sbj_list
            assert set(sbj_list) == set(sbj_ifeat_data_img), 'sbj_list error'

        feat_file_dict = next(iter(self.sbj_ifeat_data_img.values()))
        self.feat_list = sorted(feat_file_dict.keys())
        self.scale = np.eye(self.d)
        self.ref = get_ref(next(iter(feat_file_dict.values())))

        self.fs = None
        self.data = None
        self.f_data = None
        self.offset = None

        if memmap:
            # get tmp location for data
            self.f_data = tempfile.NamedTemporaryFile(suffix='.dat').name
            self.f_data = pathlib.Path(self.f_data)

        self.mask = mask
        self.__mask = None

    def discard_to(self, n_sbj, split=None):
        """ discards sbj so only num_sbj remains (in place)

        Args:
            n_sbj (int):
            split (Split): if passed, ensures at most num_sbj per split grp
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
                del self.sbj_ifeat_data_img[sbj]

        # update sbj_list
        self.sbj_list = sorted(self.sbj_ifeat_data_img.keys())

    def get_fs(self, ijk=None, mask=None, pc_ijk=None, sbj_list=None,
               sbj_bool=None):
        assert self.is_loaded, 'data_image is not loaded'
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
    def loaded(self, offset=None, **kwargs):
        """ provides context manager with loaded data

        note: we only load() and unload() if the object was previously not
        loaded
        """
        was_loaded = self.is_loaded

        # load or check that previous load was equivilent
        if was_loaded:
            if offset is None:
                # NOTE: if a data_img is loaded with an offset, future calls
                # to loaded() need not request this offset
                pass
            else:
                assert self.offset is not None, 'load() w/ offset after load()'
                assert np.isclose(offset, self.offset), 'offsets not equal'

        else:
            self.load(offset=offset, **kwargs)

        try:
            yield self
        finally:
            if not was_loaded:
                # return to original state
                self.unload()

    @contextmanager
    def data_writable(self):
        if self.memmap:
            self.data = np.array(self.data)

        try:
            yield self
        finally:
            if self.memmap:
                self._flush_memmap()

    def _flush_memmap(self):
        x = np.memmap(self.f_data, dtype='float32', mode='w',
                      shape=self.data.shape)
        x[:] = self.data[:]
        x.flush()
        self.data = np.memmap(self.f_data, dtype='float32', mode='r',
                              shape=self.data.shape)

    def reset_offset(self, offset=None):
        """ discards old offset, adds a new one (faster than reloading)

        Args:
            offset (np.array):
        """
        with self.data_writable():
            # out with the old
            if self.offset is not None:
                self.data -= self.offset

            # in with the new
            if offset is not None:
                self.data += offset
            self.offset = offset

    def load(self, offset=None, verbose=False, scale_norm=False):
        """ loads data

        Args:
            offset (np.array): image offset
            verbose (bool): toggles command line output
            scale_norm (bool): toggles scaling data to unit variance
        """

        if self.is_loaded:
            raise AttributeError('already loaded')

        # initialize array
        shape = (*self.ref.shape, self.num_sbj, self.d)
        self.data = np.empty(shape)

        # load data
        for sbj_idx, sbj in tqdm(enumerate(self.sbj_list),
                                 desc='load per sbj',
                                 disable=not verbose):
            for feat_idx, feat in enumerate(self.feat_list):
                f = self.sbj_ifeat_data_img[sbj][feat]
                img = nib.load(str(f))
                self.data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # get mask of data
        self.__mask = Mask(np.all(self.data, axis=(3, 4)), ref=self.ref)

        # get mask
        if self.mask is not None:
            self.mask = np.logical_and(self.__mask, self.mask)
        else:
            self.mask = self.__mask

        # apply offset
        if offset is not None:
            self.data += offset
        self.offset = offset

        if scale_norm:
            fs = self.get_fs(mask=self.mask)
            self.data = self.data @ np.diag((1 / np.diag(fs.cov)) ** .5)

        if self.memmap:
            self._flush_memmap()

    def unload(self):
        # delete memory map file
        if self.f_data is not None and self.f_data.exists():
            os.remove(str(self.f_data))
            self.f_data = None

        self.data = None
        self.__mask = None
        self.offset = None

    def to_nii(self, folder=None, mean=False, sbj_list=None):
        """ prints nii of each sbj's features, optionally averages across sbj

        Args:
            folder (str or Path): output folder, defaults to random tmp folder
            mean (bool): toggles averaging across sbj
            sbj_list (list): which sbj to include

        Returns:
            folder (Path): output folder
        """
        assert self.is_loaded, 'data_image is not loaded'

        def save_img(x, f):
            img = nib.Nifti1Image(x, affine=self.ref.affine)
            img.to_filename(str(folder / f))

        # get output folder, make it if need be
        if folder is None:
            folder = tempfile.TemporaryDirectory().name
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # get sbj which are to be saved
        sbj_bool = self.sbj_list_to_bool(sbj_list)

        # write to file
        for feat_idx, feat in enumerate(self.feat_list):
            if mean:
                x = self.data[:, :, :, sbj_bool, feat_idx].mean(axis=3)
                save_img(x=x, f=f'{feat}.nii.gz')
            else:
                for sbj_idx in np.where(sbj_bool)[0]:
                    x = self.data[:, :, :, sbj_idx, feat_idx]
                    sbj = self.sbj_list[sbj_idx]
                    save_img(x=x, f=f'{feat}_{sbj}.nii.gz')

        return folder

    def sbj_list_to_bool(self, sbj_list=None):
        if sbj_list is None:
            return np.ones(self.num_sbj).astype(bool)

        sbj_set = set(sbj_list)
        return np.array([sbj in sbj_set for sbj in self.sbj_list])
