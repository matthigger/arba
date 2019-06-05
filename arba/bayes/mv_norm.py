import numpy as np


class MVNorm:
    """ encapsulates bayes analysis of mvnorm with unknown mean and cov

    details identical to pg 73 of "Bayesian Data Analysis" Gelman (3rd edition)

    sig ~ InvWishart_{deg_free}(lam^-1)
    mu|sig ~ Normal(mu, sig / num_obs)

    initialize with the prior params and then call
    bayes_update() methods which return new MVNorm which have incorporated
    the observations

    Attributes:
        mu (np.array): expected mean
        lam (np.array): scaled covariance ish
        num_obs (int): number of observations
        deg_free (int): degrees of freedom
    """

    @property
    def d(self):
        return len(self.mu)

    def get_mu_marginal(self):
        """ gets params of multivariate t which describe mu's marginal

        Returns:
            deg_free (int): degrees of freedom of multivariate t
            loc (np.array): location of multivariate t
            shape (np.array): shape matrix of multivariate t
        """
        deg_free = self.deg_free - self.d + 1
        loc = self.mu
        shape = self.lam / (self.num_obs * deg_free)
        return deg_free, loc, shape

    @staticmethod
    def non_inform_dplus1(d, num_obs=1, mu=None):
        assert d > 1, 'd must be > 1'

        if mu is None:
            mu = np.zeros(d)

        return MVNorm(mu=mu,
                      lam=np.eye(d),
                      num_obs=num_obs,
                      deg_free=d + 1)

    def __init__(self, mu, lam, num_obs, deg_free):
        self.mu = np.atleast_1d(mu)
        self.lam = np.atleast_2d(lam)
        self.num_obs = int(num_obs)
        self.deg_free = int(deg_free)

    def bayes_update(self, obs_mu, obs_cov, num_obs):
        """ incorporates observations from summary statistics

        Args:
            obs_mu (np.array): observed mean
            obs_cov (np.array): observed covariance (simple average)
            num_obs (int): number of observations

        Returns:
            mv_norm (MVNorm): new MVNorm which incporates observations
        """
        deg_free = self.deg_free + num_obs
        num_obs_new = self.num_obs + num_obs

        mu_weight_self = self.num_obs / num_obs_new
        mu = self.mu * mu_weight_self + \
             obs_mu * (1 - mu_weight_self)

        c = self.num_obs * num_obs / num_obs_new
        u = obs_mu - self.mu
        lam = self.lam + num_obs * obs_cov + c * np.outer(u, u)

        return MVNorm(mu=mu, lam=lam, num_obs=num_obs_new, deg_free=deg_free)

    def bayes_update_data(self, x):
        """ incorporates observations directly

        Args:
            x (np.array): (num_obs, d) observations

        Returns:
            mv_norm (MVNorm): new MVNorm which incorporates observations
        """
        # compute mean
        obs_mu = np.mean(x, axis=0)

        # compute cov (note: ddof=0 => simple average)
        obs_cov = np.cov(x.T, ddof=0)

        return self.bayes_update(obs_mu=obs_mu, obs_cov=obs_cov,
                                 num_obs=x.shape[0])


if __name__ == '__main__':
    d = 2

    # actual
    mu_actual = np.ones(d)
    cov_actual = np.eye(d) * .01

    mv_norm = MVNorm.non_inform_dplus1(d=d)

    np.random.seed(1)
    with np.printoptions(precision=5, suppress=True):
        for rnd_idx in range(7):
            x = np.random.multivariate_normal(mean=mu_actual,
                                              cov=cov_actual,
                                              size=2 ** (rnd_idx + 2))
            mv_norm = mv_norm.bayes_update_data(x)
            print(f'round: {rnd_idx}, total obs: {mv_norm.num_obs}')
            _, mu, cov = mv_norm.get_mu_marginal()
            print(f'mu: {mu}')
            print(f'cov: {cov}\n')
