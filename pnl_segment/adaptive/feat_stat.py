import numpy as np


class FeatStat:
    """ cardinality, mean and observed variance of a set (has __add__())

    NOTE: we use the observed variance, not the unbiased estimate (see
    discussion of ddof parameter in np.cov and np.var)

    >>> a_set = range(10)
    >>> b_set = range(15)
    >>> a = FeatStat.from_iter(a_set)
    >>> b = FeatStat.from_iter(b_set)
    >>> a + b
    PointSet(n=25, mu=[[6.]], var=[[16.]])
    >>> # validation, explicitly compute via original set
    >>> FeatStat.from_iter(list(a_set) + list(b_set))
    PointSet(n=25, mu=[[6.]], var=[[16.]])
    >>> # test multi dim
    >>> np.random.seed(1)
    >>> a_array = np.random.rand(2, 10)
    >>> b_array = np.random.rand(2, 10)
    >>> a = FeatStat.from_array(a_array)
    >>> b = FeatStat.from_array(b_array)
    >>> a + b == FeatStat.from_array(np.hstack((a_array, b_array)))
    True
    """

    def __init__(self, n=0, mu=np.nan, var=np.nan):
        # defaults to 'empty' set of features
        self.n = int(n)
        self.mu = np.atleast_2d(mu)
        self.var = np.atleast_2d(var)

        if self.mu.shape[0] != self.var.shape[0]:
            if self.mu.shape[1] == self.var.shape[0]:
                self.mu = self.mu.T
            else:
                raise AttributeError('dimension mismatch: mu and var')

    def __eq__(self, other):
        if not isinstance(other, FeatStat):
            return False

        return self.n == other.n and \
               np.allclose(self.mu, other.mu) and \
               np.allclose(self.var, other.var)

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
        var = np.cov(x, ddof=0)
        return FeatStat(n=x.shape[1], mu=np.mean(x, axis=1), var=var)

    def __repr__(self):
        return f'PointSet(n={self.n}, mu={self.mu}, var={self.var})'

    def __add__(self, othr):
        if othr == 0 or othr.n == 0:
            # useful for sum(iter of PointSet), begins by adding 0 to 1st iter
            return FeatStat(self.n, self.mu, self.var)
        elif self.n == 0:
            # self is empty, return a copy of othr
            return FeatStat(othr.n, othr.mu, othr.var)

        # compute n
        n = self.n + othr.n

        # compute mu
        lam = self.n / n
        mu = self.mu * lam + \
             othr.mu * (1 - lam)

        # compute var
        var = self.n / n * (self.var + self.mu @ self.mu.T) + \
              othr.n / n * (othr.var + othr.mu @ othr.mu.T) - \
              mu @ mu.T
        return FeatStat(n, mu, var)

    __radd__ = __add__
