import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib.patches import Ellipse

sns.set(font_scale=1.2)


def plot_norm_2d(mu, cov, plot_mu=True, cdf=[.95], label=None, ax=None,
                 **kwargs):
    """ plots a 2d gaussian

    Args:
        mu (np.array): mean
        cov (np.array): cov
        plot_mu (bool): toggles whether center is plotted
        cdf (list): list of floats in (0, 1).  ellipse which contains each
                      alpha is drawn
        label (str): label
        ax (plt.Axes): axis to plot on

    Returns:
        ax (plt.Axes): axis plotted on
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    w, v = np.linalg.eig(cov)
    eig_vec0 = w[0] * v[:, 0]
    angle = np.rad2deg(np.arctan2(eig_vec0[1], eig_vec0[0]))

    def compute_draw_cdf(c):
        """ computes and draws an ellipse which contains alpha % of mass
        """

        maha = scipy.stats.chi2.ppf(c, df=len(mu))
        scale = np.sqrt(maha / w[0])

        if label is not None:
            _label = f'{label} {c:.2f}'

        # draw ellipse
        ell = Ellipse(xy=mu, angle=angle, label=_label,
                      width=w[0] * scale * 2,
                      height=w[1] * scale * 2, **kwargs)
        ax.add_artist(ell)
        return ell

    for c in cdf:
        ell = compute_draw_cdf(c)

    if plot_mu:
        plt.scatter(mu[0], mu[1], **kwargs)

    # gut check: draw and plot samples to get a sense of cdf
    x = np.random.multivariate_normal(mean=mu, cov=cov, size=100)
    plt.scatter(x[:, 0], x[:, 1], **kwargs)

    plt.grid()

    return ax, ell


if __name__ == '__main__':
    from sklearn.datasets import make_spd_matrix
    import arba.plot

    fig, ax = plt.subplots(1, 1)

    np.random.seed(1)

    kwargs = {'alpha': .3, 'ax': ax}

    _, p0 = plot_norm_2d(mu=(0, 0), cov=np.eye(2),
                         facecolor='k', label='z', **kwargs)
    _, p1 = plot_norm_2d(mu=(1, 1), cov=np.eye(2),
                         facecolor='b', label='shifted z', **kwargs)
    patch_list = [p0, p1]
    for idx in range(3):
        _cov = make_spd_matrix(2)
        _, p = plot_norm_2d(mu=(1, 1), cov=_cov,
                            facecolor='g', label=f'random{idx}', **kwargs)
        patch_list.append(p)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend(handles=patch_list)
    plt.grid(True)

    print(arba.plot.save_fig())
