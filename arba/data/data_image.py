import abc
import pathlib
import tempfile
from contextlib import contextmanager

import nibabel as nib
import numpy as np

from arba.region import FeatStat


class DataImage:
    """ manages large datasets of multivariate images

    the focus is on a context manager which loads data.  data is loaded into a
    data cube, operated on by some fncs, then memory mapped.  DataImage.data
    is then replaced with a read-only version of the memory mapped array,
    allowing for parallel processes to operate on shared memory.

    Attributes:
        sbj_list (list): list of sbj (defines indexing)
        feat_list (list): list of features (defines indexing)
        ref (RefSpace): defines shape and affine of data
        mask (np.array): mask of active area (boolean)

    Attributes available after load()
        data (np.array): (space0, space1, space2, sbj_idx, feat_idx) data
        offset (np.array): an offset which has been added to data
        f_data (Pathlib.Path): location of memmap of data.  None if not memmap
    """

    @property
    def is_memmap(self):
        return bool(self.f_data)

    @property
    def is_loaded(self):
        return self.data is not None

    @property
    def num_sbj(self):
        return len(self.sbj_list)

    @property
    def num_feat(self):
        return len(self.feat_list)

    def __init__(self, sbj_list, feat_list, ref, mask=None):
        self.sbj_list = sbj_list
        self.feat_list = feat_list
        self.ref = ref

        self.mask = mask
        if self.mask is None:
            self.mask = np.ones(ref.shape).astype(bool)

        self.data = None
        self.offset = None
        self.f_data = None

    def get_fs(self, ijk=None, mask=None, pc_ijk=None, sbj_list=None,
               sbj_bool=None):

        assert self.is_loaded, 'data_image is not loaded'
        assert not ((sbj_list is not None) and (sbj_bool is not None)), \
            'nand(sbj_list, sbj_bool) required'
        assert 2 == ((ijk is None) + (mask is None) + (pc_ijk is None)), \
            'xor(ijk, mask, pc_ijk) required'

        # get sbj_bool
        if sbj_bool is None:
            sbj_bool = self.sbj_list_to_bool(sbj_list)

        # get data array
        if ijk is not None:
            # single point
            i, j, k = ijk
            x = self.data[i, j, k, :, :]
            x = x[sbj_bool, :].reshape((-1, self.num_feat), order='F')
        elif mask is not None:
            # mask
            x = self.data[mask, :, :]
            x = x[:, sbj_bool, :].reshape((-1, self.num_feat), order='F')
        else:
            # point cloud
            n = len(pc_ijk)
            x = np.empty((n, self.num_feat))
            for idx, (i, j, k) in enumerate(pc_ijk):
                _x = self.data[i, j, k, :, :]
                x[idx, :] = _x[sbj_bool, :].reshape((-1, self.num_feat), order='F')

        return FeatStat.from_array(x.T)

    @contextmanager
    def loaded(self, offset=None, **kwargs):
        """ context manager which ensures data is loaded into object

        this context manager may be nested without error
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
        if self.is_memmap:
            self.data = np.array(self.data)

        try:
            yield self
        finally:
            if self.is_memmap:
                self.flush_to_memmap()

    def flush_to_memmap(self):
        assert self.f_data is None, 'cannot flush until last memmap deleted'

        self.f_data = tempfile.NamedTemporaryFile(suffix='.dat').name
        self.f_data = pathlib.Path(self.f_data)

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

    @abc.abstractmethod
    def load(self, _data, offset=None, memmap=False):
        """ loads data

        Args:
            _data (np.array): data to be loaded
            offset (np.array): image offset
            memmap (bool): toggles whether array is written to memory map
        """
        # save
        self.data = _data

        # apply offset
        if offset is not None:
            self.data += offset
        self.offset = offset

        # apply mask
        self.data[np.logical_not(self.mask)] = 0

        # memmap
        if memmap:
            self.flush_to_memmap()

    @abc.abstractmethod
    def unload(self):
        if self.memmap:
            self.f_data.unlink(missing_ok=True)
        self.data = None
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
