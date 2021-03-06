import numpy as np
from scipy.stats import multivariate_normal


class FeatStat:
    """ cardinality, mean and observed variance of a set (has __add__())

    NOTE: we use the observed variance, not the unbiased estimate (see
    discussion of ddof parameter in np.cov and np.cov)

    >>> a_set = range(10)
    >>> b_set = range(15)
    >>> a = FeatStat.from_iter(a_set)
    >>> b = FeatStat.from_iter(b_set)
    >>> a + b
    FeatStat(n=25, mu=[6.], cov=[[16.]])
    >>> # validation, explicitly compute via original set
    >>> FeatStat.from_iter(list(a_set) + list(b_set))
    FeatStat(n=25, mu=[6.], cov=[[16.]])
    >>> # test multi dim
    >>> np.random.seed(1)
    >>> a_array = np.random.rand(2, 10)
    >>> b_array = np.random.rand(2, 10)
    >>> a = FeatStat.from_array(a_array)
    >>> b = FeatStat.from_array(b_array)
    >>> a + b == FeatStat.from_array(np.hstack((a_array, b_array)))
    True
    """

    @property
    def d(self):
        return self.mu.size

    @property
    def cov_det(self):
        if self.__cov_det is None:
            self.__cov_det = np.linalg.det(self.cov)
        return self.__cov_det

    @property
    def cov_inv(self):
        """ inverse of covariance, defaults to pseudoinverse if singular
        """
        if self.__cov_inv is None:
            try:
                self.__cov_inv = np.linalg.inv(self.cov)
            except np.linalg.linalg.LinAlgError:
                self.__cov_inv = np.linalg.pinv(self.cov)
        return self.__cov_inv

    @property
    def outer_mu(self):
        if self._outer_mu is None:
            self._outer_mu = np.outer(self.mu, self.mu)

        return self._outer_mu

    def __init__(self, n, mu, cov, _fast=False, outer_mu=None):
        # defaults to 'empty' set of features
        self.n = n
        self.mu = mu
        self.cov = cov
        self.__cov_det = None
        self.__cov_inv = None
        self._outer_mu = outer_mu

        if _fast:
            return

        self.n = int(self.n)
        self.mu = np.atleast_1d(self.mu)
        self.cov = np.atleast_2d(self.cov)

        assert self.n > 0, 'n must be positive'
        assert len(self.mu.shape) == 1, 'mu must be 1d'
        assert np.allclose(self.cov, self.cov.T), 'non symmetric covariance'
        assert (np.diag(self.cov) >= 0).all(), 'negative covariance'
        assert self.d == self.cov.shape[0] == self.cov.shape[1], \
            'dim mismatch: mu and covar'

    def __eq__(self, other):
        if not isinstance(other, FeatStat):
            return False

        return self.n == other.n and \
               np.allclose(self.mu, other.mu) and \
               np.allclose(self.cov, other.cov)

    def to_normal(self, bessel=True):
        """ outputs max likelihood multivariate_normal based on features

        Args:
            bessel (bool): toggles bessel correction (for unbiased cov)

        Returns:
            rv (multivariate_normal): normal distribution
        """
        bessel = (self.n / (self.n - 1)) ** bessel
        return multivariate_normal(mean=np.squeeze(self.mu),
                                   cov=self.cov * bessel)

    @staticmethod
    def from_iter(x):
        x = np.fromiter(iter(x), dtype=type(next(iter(x))))
        return FeatStat.from_array(np.atleast_2d(x))

    @staticmethod
    def from_array(x, obs_greater_dim=None, _fast=False):
        """

        Args:
            x (np.array): observations
            obs_greater_dim (bool): "num observations > num dimensions", if
                                    true ensures observations are greater than
                                    num dim.  otherwise ensures opposite.
                                    (consider no difference in input between
                                    2x observations of a scalar and 1x
                                    observation of a 2d feature)
        """
        x = np.atleast_2d(x)
        if obs_greater_dim is not None:
            if obs_greater_dim and x.shape[0] > x.shape[1] or \
                    not obs_greater_dim and x.shape[1] > x.shape[0]:
                x = x.T

        n = x.shape[1]
        if n == 0:
            return FeatStatEmpty()
        elif n == 1:
            return FeatStatSingle(mu=np.mean(x, axis=1))

        cov = np.cov(x, ddof=0)
        assert (np.diag(
            np.atleast_2d(cov)) >= 0).all(), 'non positive covariance'
        fs = FeatStat(n=n, mu=np.mean(x, axis=1), cov=cov, _fast=False)
        return fs

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(n={self.n}, mu={self.mu}, cov={self.cov})'

    def __add__(self, othr):
        if othr == 0 or othr.n == 0:
            # useful for sum(iter of PointSet), begins by adding 0 to 1st iter
            return FeatStat(self.n, self.mu, self.cov)
        elif self.n == 0:
            # self is empty, return a copy of othr
            return FeatStat(othr.n, othr.mu, othr.cov)

        # compute n
        n = self.n + othr.n
        lambda_self = self.n / n
        lambda_othr = othr.n / n

        # compute mu
        mu = self.mu * lambda_self + \
             othr.mu * lambda_othr
        outer_mu = np.outer(mu, mu)

        # compute cov
        cov = lambda_self * (self.cov + self.outer_mu) + \
              lambda_othr * (othr.cov + othr.outer_mu)
        cov -= outer_mu

        return FeatStat(n, mu, cov, _fast=True, outer_mu=outer_mu)

    __radd__ = __add__

    def __sub__(self, othr):

        lambda_othr = othr.n / self.n
        lambda_out = 1 - lambda_othr
        n = self.n - othr.n
        assert n > 0, 'invalid subtraction: not enough observations in other'

        mu = (self.mu - (lambda_othr * othr.mu)) / lambda_out
        outer_mu = np.outer(mu, mu)

        cov = ((self.cov + self.outer_mu) -
               (othr.cov + othr.outer_mu) * lambda_othr)
        cov *= 1 / lambda_out
        cov -= outer_mu

        return FeatStat(n, mu, cov, _fast=True, outer_mu=outer_mu)

    def __reduce_ex__(self, *args, **kwargs):
        self.reset()
        return super().__reduce_ex__(*args, **kwargs)

    def reset(self):
        self.__cov_det = None
        self.__cov_inv = None

    def scale(self, a):
        """ NOTE: left multiply"""
        mu = a @ self.mu
        cov = a @ self.cov @ a.T
        return FeatStat(n=self.n, mu=mu, cov=cov)

    @staticmethod
    def get_pool_cov(fs_list):
        """ compute pooled covariance from list of FeatStat"""
        pool_cov = 0
        n = sum(fs.n for fs in fs_list)
        for fs in fs_list:
            pool_cov += fs.cov * (fs.n / n)
        return pool_cov


class FeatStatSingle(FeatStat):
    """ minimizes storage if a FeatStat of a single observation is needed

    this object interacts with FeatStat as needed while

    >>> a_set = range(10)
    >>> FeatStat.from_iter(a_set)
    FeatStat(n=10, mu=[[4.5]], cov=[[8.25]])
    >>> fs_single_list = [FeatStatSingle(feat) for feat in a_set]
    >>> fs_single_list[3]
    FeatStatSingle(n=1, mu=[[3]], cov=[[0.]])
    >>> sum(fs_single_list)
    FeatStat(n=10, mu=[[4.5]], cov=[[8.25]])
    """
    n = 1
    __cov_det = np.nan
    __cov_inv = np.nan

    @property
    def cov(self):
        d = self.d
        return np.zeros((d, d))

    def __init__(self, mu, n=1, cov=None, _outer_mu=None):
        if n != 1 or cov is not None:
            raise AttributeError
        self.mu = np.atleast_1d(mu)
        self._outer_mu = _outer_mu


class FeatStatEmpty(FeatStat):
    n = 0
    d = np.nan
    mu = np.nan
    cov = np.nan
    _outer_mu = None

    def __init__(self, *args, **kwargs):
        pass
