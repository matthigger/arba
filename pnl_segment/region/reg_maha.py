from collections import namedtuple

import numpy as np
from scipy.stats import f

from .reg import Region

# save a few essential stats
MementoMaha = namedtuple('MementoMaha', ('hash', 'size', 'maha', 'pval'))


class RegionMaha(Region):
    """ computes MSE error of representing maha per voxel by a single value

    Attributes:
        maha_per_vox (np.array): objective function per each voxel
    """

    def get_memento(self):
        # todo: this whole thing should be replaced with a simple hash ...
        return MementoMaha(hash=hash(self),
                           size=len(self),
                           maha=self.maha,
                           pval=self.pval)

    def __init__(self, *args, maha_per_vox=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._maha = None
        self._pval = None

        if maha_per_vox is None:
            self.maha_per_vox = np.array(self.maha)
        else:
            self.maha_per_vox = np.array(maha_per_vox)

        if len(self.maha_per_vox.shape) > 1:
            raise AttributeError('maha_per_vox shape isnt 1d')

    @property
    def maha(self):
        # memoize
        if self._maha is None:
            self._maha = self.get_maha()
        return self._maha

    def get_maha(self):
        """ negative Mahalanobis squared between active_grp

        maha(p_1, p_0) = (u_1 - u_0)^T sig ^ -1 (u_1 - u_0)

        where sig is the common covariance / region size (in voxels)
        """

        fs0, fs1 = self.fs_dict.values()

        mu_diff = fs0.mu - fs1.mu
        cov = (fs0.n * fs0.cov + fs1.n * fs1.cov) / (fs0.n + fs1.n - 2)
        return mu_diff @ np.linalg.inv(cov) @ mu_diff

    @property
    def pval(self):
        if self._pval is None:
            self._pval = self.get_pval()
        return self._pval

    def get_pval(self, maha=None):
        """ gets pvalue of mahalanobis

        see http://math.bme.hu/~marib/tobbvalt/tv5.pdf

        Args:
            maha (float): mahalanobis used in computation, defaults to
                          self.maha (helpful for evaluating distribution)
        """
        if maha is None:
            maha = self.maha

        # dimensionality of data
        p = next(iter(self.fs_dict.values())).d

        # number of observations per group
        n, m = [fs.n for fs in self.fs_dict.values()]

        # compute f stat
        f_stat = maha * n * m / (n + m)
        f_stat *= (n + m - p - 1) / (p * (n + m - 2))

        # compute pval
        pval = f.sf(f_stat, dfd=p, dfn=n + m - p - 1)

        return pval

    @property
    def sq_error(self):
        """ squared error of maha per voxel represented as whole region maha
        """
        d = self._maha - self.maha_per_vox
        return np.inner(d, d)

    @staticmethod
    def get_error(reg_1, reg_2, reg_union=None):
        if reg_union is None:
            reg_union = reg_1 + reg_2
        return reg_union.sq_error - reg_1.sq_error - reg_2.sq_error

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            reg_out = super().__add__(other)
            reg_out.obj_per_vox = self.maha_per_vox
            return reg_out

        if not isinstance(other, type(self)):
            raise TypeError

        reg_out = super().__add__(other)
        reg_out.obj_per_vox = np.hstack(
            (self.maha_per_vox, other.maha_per_vox))

        if len(self.maha_per_vox.shape) > 1:
            raise AttributeError('maha_per_vox shape isnt 1d')

        return reg_out

    __radd__ = __add__

    def reset(self):
        self._maha = None
        self._pval = None
