import pathlib
import tempfile
from string import ascii_uppercase

import numpy as np

import arba.space
from .data_image import DataImage


class DataImageSynth(DataImage):
    """ builds gaussian noise nii in temp location (for testing purposes)

    features are labelled alphabetically ('feat_imgA', 'feat_imgB', ....)
    sbj are labelled numerically ('sbj0', 'sbj1', ...)
    """
    # DataImageSynth are always loaded
    is_loaded = True

    # todo: this attribute not used by subclass ... breaks abstraction
    sbj_ifeat_data_img = None

    @staticmethod
    def get_sbj_list(n):
        """ builds dummy list of sbj"""
        w = np.ceil(np.log10(n)).astype(int)
        return [f'sbj{str(idx).zfill(w)}' for idx in range(n)]

    @staticmethod
    def get_feat_list(n):
        """ builds dummy list of feat"""
        return [f'feat_img{x}' for x in ascii_uppercase[:n]]

    def __init__(self, data, sbj_list=None, feat_list=None, memmap=False,
                 mask=None):
        """ given a data matrix, writes files to nii and returns DataImage

        Args:
            data (np.array): (shape0, shape1, shape2, num_sbj, num_feat)
            sbj_list (list): list of sbj
            feat_list (list): list of features

        Returns:
            data_img (DataImageSynth)
        """

        assert len(data.shape) == 5, 'data must be of dimension 5'

        # sbj_list, feat_list
        num_sbj, num_feat = data.shape[3:]
        self.sbj_list = sbj_list
        if self.sbj_list is None:
            self.sbj_list = self.get_sbj_list(num_sbj)
        self.feat_list = feat_list
        if self.feat_list is None:
            self.feat_list = self.get_feat_list(num_feat)

        # store data
        self.data = data
        self.offset = None

        # build dummy ref space
        self.ref = arba.space.RefSpace(shape=data.shape[:3])

        # mask
        self.mask = mask
        if self.mask is None:
            self.mask = np.ones(self.ref.shape).astype(bool)
        else:
            self.data[np.logical_not(self.mask)] = 0

        # memmap
        self.f_data = None
        if memmap:
            # get tmp location for data
            self.f_data = tempfile.NamedTemporaryFile(suffix='.dat').name
            self.f_data = pathlib.Path(self.f_data)
        if self.memmap:
            self._flush_memmap()

    def load(self):
        """ DataImageSynth have no files, can't be loaded """
        pass

    def unload(self):
        """ DataImageSynth have no files, can't be unloaded """
        pass
