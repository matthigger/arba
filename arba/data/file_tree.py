import pathlib
import tempfile

import nibabel as nib
import numpy as np

from ..region import FeatStat
from ..space import get_ref, Mask, PointCloud


def check_loaded(fnc):
    def wrapped(self, *args, **kwargs):
        assert self.data is not None, 'FileTree must be loaded to call'
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
        pc (PointCloud): point cloud of active area
        fs (FeatStat): feature statistics across active area of all sbj
        data (np.memmap): if data is loaded, a read only memmap of data
    """

    @property
    def num_sbj(self):
        return len(self.sbj_feat_file_tree)

    @property
    def d(self):
        return len(self.feat_list)

    def __len__(self):
        return len(self.sbj_feat_file_tree.keys())

    def __init__(self, sbj_feat_file_tree, fnc_list=None):
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())
        feat_file_dict = next(iter(self.sbj_feat_file_tree.values()))
        self.feat_list = sorted(feat_file_dict.keys())
        self.scale = np.eye(self.d)
        self.ref = get_ref(next(iter(feat_file_dict.values())))

        self.fnc_list = fnc_list
        if fnc_list is None:
            self.fnc_list = list()

        self.mask = None
        self.pc = None
        self.fs = None
        self.data = None
        self.f_data = None

    def __enter__(self):
        """ loads data, applies fnc in self.fnc_list
        """
        # build memmap file
        shape = (*self.ref.shape, self.num_sbj, self.d)
        self.data = np.zeros(shape)

        # load data
        for sbj_idx, sbj in enumerate(self.sbj_list):
            for feat_idx, feat in enumerate(self.feat_list):
                f = self.sbj_feat_file_tree[sbj][feat]
                img = nib.load(str(f))
                self.data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # get pc, mask
        self.mask = Mask(np.all(self.data, axis=(3, 4)), ref=self.ref)
        self.pc = PointCloud.from_mask(self.mask)

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
        return self

    def __exit__(self, err_type, value, traceback):
        # reset attributes which must be loaded
        self.mask = None
        self.pc = None
        self.fs = None
        self.data = None
        self.f_data = None

        # delete memory map file

    @check_loaded
    def get_data_subset(self, sbj_list=None):
        """ returns view of data corresponding only to sbj in sbj_list
        """
        raise NotImplementedError

    @check_loaded
    def to_nii(self, fnc=None, feat=None, sbj_list=None, f_out=None,
               back=0):
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
        sbj_bool = self.sbj_list_to_bool(sbj_list)

        # build img
        if fnc is not None:
            # build based on custom fnc
            x = back * np.ones(self.ref.shape)
            for i, j, k in self.pc:
                x[i, j, k] = fnc(self.data[i, j, k, sbj_bool, :])
        else:
            # build based on mean feature
            feat_idx = self.feat_list.index(feat)
            x = np.mean(self.data[:, :, :, sbj_bool, feat_idx], axis=3)

        # write to file
        img = nib.Nifti1Image(x, affine=self.ref.affine)
        img.to_filename(str(f_out))

        return pathlib.Path(f_out)

    def sbj_list_to_bool(self, sbj_list=None):
        if sbj_list is None:
            return np.ones(self.num_sbj).astype(bool)

        sbj_set = set(sbj_list)
        return np.array(sbj in sbj_set for sbj in self.sbj_list)

    def sbj_bool_to_list(self, sbj_bool):
        return [sbj for b, sbj in zip(sbj_bool, self.sbj_list) if b]


def scale_normalize(ft):
    """ compute & store mean and var per feat, scale + offset to Z score

    Args:
        ft (FileTree): file tree to be equalized
    """
    # compute stats
    shape = (len(ft.pc) * ft.num_sbj, ft.d)
    shape_orig = (len(ft.pc), ft.num_sbj, ft.d)
    _data = ft.data[ft.mask, :, :].reshape(shape, order='F')
    ft.fs = FeatStat.from_array(_data.T)

    # apply scale equalization
    scale = np.diag(1 / np.diag(ft.fs.cov)) ** .5
    _data = (_data - ft.fs.mu) @ scale
    ft.data[ft.mask, :, :] = _data.reshape(shape_orig, order='F')
