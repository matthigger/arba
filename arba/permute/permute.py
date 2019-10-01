import pathlib
import tempfile
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from mh_pytools import parallel


class Permute:
    """ runs permutation testing to find regions whose stat is significant

    additionally, this object serves as a container for the result objects

    Attributes:
        num_perm (int): number of permutations to run
        alpha (float): confidence threshold
        data_img (DataImage): observed imaging data
    """

    def __init__(self, data_img, alpha=.05, num_perm=100, mask_target=None,
                 verbose=True, folder=None, par_flag=False):
        assert alpha >= 1 / (num_perm + 1), \
            'not enough perm for alpha, never sig'

        self.alpha = alpha
        self.num_perm = num_perm
        self.data_img = data_img
        self.mask_target = mask_target
        self.verbose = verbose

        self.sg_hist = None
        self.merge_record = None

        self.folder = folder
        if self.folder is None:
            self.folder = pathlib.Path(tempfile.mkdtemp())

        with data_img.loaded():
            self.run_single()

            self.permute(par_flag=par_flag)

    @abstractmethod
    def _set_seed(self, seed=None):
        raise NotImplementedError

    @abstractmethod
    def get_sg_hist(self, seed=None):
        """ constructs a SegGraphHistory """
        raise NotImplementedError

    @abstractmethod
    def run_single_permute(self, seed):
        raise NotImplementedError

    def run_single(self):
        """ runs a single Agglomerative Clustering run
        """
        # build sg_hist, reduce
        self.sg_hist = self.get_sg_hist()
        self.sg_hist.reduce_to(1, verbose=self.verbose)
        self.merge_record = self.sg_hist.merge_record

    @abstractmethod
    def permute(self, par_flag=False):
        # if seed = 0, evaluates as false and doesn't do anything
        seed_list = np.arange(1, self.num_perm + 1)
        arg_list = [{'seed': x} for x in seed_list]

        if par_flag:
            val_list = parallel.run_par_fnc(obj=self, fnc='run_single_permute',
                                            arg_list=arg_list,
                                            verbose=self.verbose)
        else:
            val_list = list()
            for d in tqdm(arg_list, desc='permute', disable=not self.verbose):
                val_list.append(self.run_single_permute(**d))

        # reset permutation to original data
        self._set_seed(seed=None)

        return val_list
