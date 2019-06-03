import numpy as np
import pymc3 as pm

from arba.plot import save_fig

SEED = 1

np.random.seed(SEED)

num_sbj = 3
obs_per_sbj = 100
cores = 20
d = 2

gt_mu = np.zeros(d)
gt_sig_sbj = np.eye(d)

gt_mu_sbj = np.random.multivariate_normal(gt_mu, gt_sig_sbj, size=num_sbj)
gt_sig_space = np.eye(d) * .2

obs = np.ones((obs_per_sbj, d, num_sbj)) * np.nan
for sbj_idx, _mu in enumerate(gt_mu_sbj):
    _obs = np.random.multivariate_normal(_mu, gt_sig_space,
                                         size=obs_per_sbj)
    obs[..., sbj_idx] = _obs

mean_obs = obs.mean(axis=(0, 2))

with pm.Model() as model:
    packed_sig_sbj = pm.LKJCholeskyCov('packed_sig_sbj', n=d,
                                       eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
    L_sbj = pm.expand_packed_triangular(d, packed_sig_sbj)
    sig_sbj = pm.Deterministic('sig_sbj', L_sbj.dot(L_sbj.T))

    packed_sig_space = pm.LKJCholeskyCov('packed_sig_space', n=d,
                                         eta=2.,
                                         sd_dist=pm.HalfCauchy.dist(2.5))
    L_space = pm.expand_packed_triangular(d, packed_sig_sbj)
    sig_space = pm.Deterministic('sig_space', L_sbj.dot(L_sbj.T))

    mu = pm.Normal('mu', 0., 10., shape=2, testval=mean_obs)

    mu_sbj_list = list()
    obs_sbj_list = list()
    for sbj_idx in range(num_sbj):
        _mu = pm.Normal(f'mu_{sbj_idx}', 0., 10., shape=2, testval=0)
        _obs = pm.MvNormal(f'obs_{sbj_idx}', mu, chol=L_space,
                           observed=obs[..., sbj_idx])
        mu_sbj_list.append(_mu)
        obs_sbj_list.append(_obs)

    trace = pm.sample(random_seed=SEED, cores=cores)

pm.plot_posterior(trace, varnames=['mu'])
print(save_fig())
