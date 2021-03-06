import nibabel as nib
import numpy as np
from tqdm import tqdm

from arba.space import Mask, get_ref
from .data_image import DataImage


class DataImageFile(DataImage):
    """ manages large datasets of multivariate images (from nii)

    Attributes:
        sbj_ifeat_data_img (tree): key0 is sbj, key2 is imaging feat, vals are
                                    files which contain imaging feat of sbj

        mask_init (np.array): mask argument passed by user
        mask_img (np.array): mask where all vox are non zero (across sbj-feat)
        mask (np.array): logical and of two masks above
    """

    def __init__(self, sbj_ifeat_data_img, *args, **kwargs):
        self.sbj_ifeat_data_img = sbj_ifeat_data_img
        sbj_list = sorted(self.sbj_ifeat_data_img.keys())
        ifeat_data_img_dict = next(iter(self.sbj_ifeat_data_img.values()))
        feat_list = sorted(ifeat_data_img_dict.keys())
        ref = get_ref(next(iter(ifeat_data_img_dict.values())))
        super().__init__(*args, sbj_list=sbj_list, feat_list=feat_list,
                         ref=ref, **kwargs)

        self.mask_init = self.mask
        self.mask_img = None

    def load(self, *args, verbose=False, **kwargs):
        """ loads data from files into self.data

        Args:
            verbose (bool): toggles command line output
        """

        if self.is_loaded:
            raise AttributeError('already loaded')

        # initialize array
        shape = (*self.ref.shape, self.num_sbj, self.num_feat)
        data = np.empty(shape)

        # load data
        for sbj_idx, sbj in tqdm(enumerate(self.sbj_list),
                                 desc='load per sbj',
                                 disable=not verbose):
            for feat_idx, feat in enumerate(self.feat_list):
                f = self.sbj_ifeat_data_img[sbj][feat]
                img = nib.load(str(f))
                data[:, :, :, sbj_idx, feat_idx] = img.get_data()

        # get mask of data
        self.mask_img = Mask(np.all(data, axis=(3, 4)), ref=self.ref)

        # get mask
        if self.mask_init is None:
            self.mask = self.mask_img
        else:
            self.mask = np.logical_and(self.mask_img, self.mask_init)

        super().load(*args, _data=data, **kwargs)

    def unload(self):
        self.mask_img = None
        self.mask = self.mask_init
        super().unload()
