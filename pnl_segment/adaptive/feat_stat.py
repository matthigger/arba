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
    PointSet(n=25, mu=[[6.]], cov=[[16.]])
    >>> # validation, explicitly compute via original set
    >>> FeatStat.from_iter(list(a_set) + list(b_set))
    PointSet(n=25, mu=[[6.]], cov=[[16.]])
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

    def __init__(self, n=0, mu=np.nan, cov=np.nan, label=None):
        # defaults to 'empty' set of features
        self.n = int(n)
        self.mu = np.atleast_2d(mu)
        self.cov = np.atleast_2d(cov)
        self.label = label

        self.__cov_det = None
        self.__cov_inv = None

        if self.mu.shape[0] != self.cov.shape[0]:
            if self.mu.shape[1] == self.cov.shape[0]:
                self.mu = self.mu.T
            else:
                raise AttributeError('dimension mismatch: mu and cov')

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
    def from_array(x, obs_greater_dim=True):
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
        if obs_greater_dim and x.shape[0] > x.shape[1] or \
                not obs_greater_dim and x.shape[1] > x.shape[0]:
            x = x.T
        cov = np.cov(x, ddof=0)
        return FeatStat(n=x.shape[1], mu=np.mean(x, axis=1), cov=cov)

    def __repr__(self):
        return f'FeatStat(n={self.n}, mu={self.mu}, cov={self.cov})'

    def __add__(self, othr):
        if othr == 0 or othr.n == 0:
            # useful for sum(iter of PointSet), begins by adding 0 to 1st iter
            return FeatStat(self.n, self.mu, self.cov)
        elif self.n == 0:
            # self is empty, return a copy of othr
            return FeatStat(othr.n, othr.mu, othr.cov)

        # compute n
        n = self.n + othr.n

        # compute mu
        lam = self.n / n
        mu = self.mu * lam + \
             othr.mu * (1 - lam)

        # compute cov
        var = self.n / n * (self.cov + self.mu @ self.mu.T) + \
              othr.n / n * (othr.cov + othr.mu @ othr.mu.T) - \
              mu @ mu.T

        # note: no checking for label, we take it from self
        return FeatStat(n, mu, var, label=self.label)

    __radd__ = __add__
