import pathlib
import tempfile

import nibabel as nib
import numpy as np

from ..space import get_ref, Mask, PointCloud


def check_loaded(fnc):
    def wrapped(self, *args, **kwargs):
        assert self.f_data is not None, 'FileTree must be loaded to call'
        return fnc(*args, **kwargs)

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

    def __init__(self, sbj_feat_file_tree):
        self.sbj_feat_file_tree = sbj_feat_file_tree
        self.sbj_list = sorted(self.sbj_feat_file_tree.keys())
        feat_file_dict = next(iter(self.sbj_feat_file_tree.values()))
        self.feat_list = sorted(feat_file_dict.keys())
        self.scale = np.eye(self.d)
        self.ref = get_ref(next(iter(feat_file_dict.values())))

        self.mask = None
        self.pc = None
        self.fs = None
        self.data = None
        self.f_data = None

    def load(self, fnc_list=None):
        """ loads data

        Args:
            fnc_list (list): each is a function which is passed data.  allows
                             modification of data before it is write protected
                             (see self.scale_equalize() for example call sig)
        """
        # build memmap file
        shape = (*self.ref.shape, self.num_sbj, self.d)
        self.f_data = tempfile.NamedTemporaryFile(suffix='.dat').name
        self.f_data = pathlib.Path(self.f_data)
        data = np.memmap(self.f_data, dtype='float32', mode='w+', shape=shape)
        data[:] = 0

        # load data
        for sbj_idx, sbj in enumerate(self.sbj_list):
            for feat_idx, feat in enumerate(self.feat_list):
                f = self.sbj_feat_file_tree[sbj][feat]
                img = nib.load(str(f))
                data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # get pc, mask
        self.mask = Mask(np.all(data, axis=(3, 4)), ref=self.ref)
        self.pc = PointCloud.from_mask(self.mask)

        # apply all fnc
        if fnc_list is not None:
            for fnc in fnc_list:
                fnc(data)

        # flush data to memmap, make read only copy
        del data
        self.data = np.memmap(self.f_data, dtype='float32', mode='r',
                              shape=shape)

        return self

    def __enter__(self, *args, **kwargs):
        self.load(*args, **kwargs)
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
    def scale_equalize(self):
        """ compute & store mean and var per feat, scale + offset to Z score
        """
        # compute features

        # apply scale equalization
        raise NotImplementedError

    @check_loaded
    def to_nii(self, fnc=None, feat=None, sbj_list=None, f_out=None):
        """ writes data to a nii file

        Args:
            fnc (fnc): accepts data array, returns a 3d volume
            feat : some feature in feat_list, if passed, fnc returns mean feat
            sbj_list (list): list of sbj to include, if None all sbj used
            f_out (str or Path): output nii file

        Returns:
            f_out (Path): nii file out
        """
        raise NotImplementedError
