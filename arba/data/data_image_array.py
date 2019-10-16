from string import ascii_uppercase

import numpy as np

import arba.space
from .data_image import DataImage


class DataImageArray(DataImage):
    """ DataImage from an array
    """
    @staticmethod
    def get_sbj_list(n):
        """ builds dummy list of sbj"""
        w = np.ceil(np.log10(n)).astype(int)
        return [f'sbj{str(idx).zfill(w)}' for idx in range(n)]

    @staticmethod
    def get_feat_list(n):
        """ builds dummy list of feat"""
        return [f'feat_img{x}' for x in ascii_uppercase[:n]]

    def __init__(self, data, sbj_list=None, feat_list=None, **kwargs):
        # sbj_list, feat_list
        num_sbj, num_feat = data.shape[3:]
        if sbj_list is None:
            sbj_list = self.get_sbj_list(num_sbj)
        if feat_list is None:
            feat_list = self.get_feat_list(num_feat)

        ref = arba.space.RefSpace(shape=data.shape[:3])

        super().__init__(sbj_list=sbj_list, feat_list=feat_list, ref=ref,
                         **kwargs)
        super().load(_data=data)

    def load(self):
        """ DataImageArray have no files, can't be loaded """
        pass

    def unload(self):
        """ DataImageArray have no files, can't be unloaded """
        pass
