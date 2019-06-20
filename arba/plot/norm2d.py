import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib.patches import Ellipse

sns.set(font_scale=1.2)


def plot_norm_2d(mu, cov, plot_mu=True, cdf=[.95], label=None, ax=None,
                 reset_lim=True, **kwargs):
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
    else:
        plt.sca(ax)

    w, v = np.linalg.eig(cov)
    eig_vec0 = w[0] * v[:, 0]
    angle = np.rad2deg(np.arctan2(eig_vec0[1], eig_vec0[0]))

    def compute_draw_cdf(c):
        """ computes and draws an ellipse which contains alpha % of mass
        """
        # compute target mahalanobis
        maha = scipy.stats.chi2.ppf(c, df=len(mu))

        # compute vector scaling (note: maha of eig vec is eig value)
        scale_width = np.sqrt(maha / w[0])
        scale_height = np.sqrt(maha / w[1])

        if label is not None:
            _label = f'{label} {c:.2f}'

        # draw ellipse
        ell = Ellipse(xy=mu, angle=angle, label=_label,
                      width=w[0] * scale_width * 2,
                      height=w[1] * scale_height * 2, **kwargs)
        ax.add_artist(ell)
        return ell

    for c in cdf:
        ell = compute_draw_cdf(c)

    if plot_mu:
        plt.scatter(mu[0], mu[1], **kwargs)

    plt.sca(ax)
    plt.grid(True)

    return ax, ell


if __name__ == '__main__':
    from sklearn.datasets import make_spd_matrix
    import arba.plot

    fig, ax = plt.subplots(1, 1)

    np.random.seed(1)

    for idx in range(3):
        _cov = make_spd_matrix(2)
        _, p = plot_norm_2d(mu=(0, 0), cov=_cov,
                            facecolor='g', label=f'random{idx}', alpha=.5)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend(handles=[p])
        plt.grid(True)

        print(arba.plot.save_fig())
