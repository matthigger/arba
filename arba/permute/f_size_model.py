import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


class FSizeModel:
    """ models  how F stats vary with region size, gets size adjusted F (saf)

    Attributes:
        size (np.array): sorted array of observed sizes
        self.logf_train (np.array): (num_obs, num_size) log f stats used to
                                    train model
        logf_expect (np.array): f_expect[idx] is the expected f stat for size
        std_expect (np.array): std_expect[idx] is the std dev of f stat of
                                size voxels
    """

    @staticmethod
    def from_list_dict(list_dict):
        """ initializes from a list of dicts

        Args:
            list_dict (list): each element is a dict with values 'size' and 'f'
        """
        size = np.vstack(d['size'] for d in list_dict)
        size = np.sort(np.unique(size))

        num_obs = len(list_dict)
        f = np.empty((num_obs, len(size))) * np.nan
        for obs_idx, d in enumerate(list_dict):
            idx = np.searchsorted(size, d['size'])
            f[obs_idx, idx] = d['f']

        return FSizeModel(size=size, f=f)

    def __init__(self, size, f):
        """ builds model

        Args:
            size (np.array): (num_size) sizes observed
            f (np.array): (num_obs, num_size) each row is a set of max f stats
                          per each each (0 if unobserved in permutation)
        """
        self.size = size.astype(int)
        self.logf_train = make_monotonic_up(np.log10(f))
        self.logf_expect = np.median(self.logf_train, axis=0)
        err = self.logf_train - self.logf_expect
        self.std_expect = np.mean(np.abs(err), axis=0)

    def get_logf_std(self, size):
        """ returns logf expected and std dev for a given size

        not all sizes are observed. if between size given then its linearly
        inerpolated

        Args:
            size (int): size of region

        Returns:
            logf (float): expected log of f stat
            std (flaot): standard deviation around the expected value
        """

        # find insert point of size
        idx = np.searchsorted(self.size, size)

        # if exact match found, return it
        if self.size[idx] == size:
            return self.logf_expect[idx], self.std_expect[idx]

        # compute weight the left (0) and right (1) sizes
        size_l, size_r = np.log10(self.size[idx: idx + 2])
        lam_r = (np.log10(size) - size_l) / (size_r - size_l)
        lam_l = 1 - lam_r

        # update
        logf = lam_l * self.logf_expect[idx] + \
               lam_r * self.logf_expect[idx + 1]
        std = lam_l * self.std_expect[idx] + \
              lam_r * self.std_expect[idx + 1]

        return logf, std

    def get_saf(self, size, f_stats, log_flag=True):
        """ returns a size adjusted f stat

        Args:
            size (np.array): sizes of f statistics
            f_stats (np.array): f stats

        Returns:
            saf (np.array): size adjusted f stat
        """
        assert np.array_equal(size.shape, f_stats.shape), 'shape mismatch'

        if log_flag:
            logf = np.log10(f_stats)
        else:
            logf = np.array(f_stats)

        fas = np.empty(f_stats.shape)
        for idx, (_size, _logf) in enumerate(zip(size.flatten(),
                                                 logf.flatten())):
            logf, std = self.get_logf_std(_size)
            fas[idx] = (_logf - logf) / std

        return fas

    def plot_model(self, ax=None):
        """ plots model """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            plt.sca(ax)

        ax.set_yscale('log')
        ax.set_xscale('log')

        for _logf in self.logf_train:
            h = plt.plot(self.size, 10 ** _logf, alpha=.4, color='k')

        mu = 10 ** self.logf_expect
        top = 10 ** (self.logf_expect + self.std_expect)
        btm = 10 ** (self.logf_expect - self.std_expect)

        h_model = plt.plot(self.size, mu, color='b', linewidth=3)
        h_model_fill = plt.fill_between(self.size, top, btm,
                                        color='b', linewidth=3, alpha=.5)
        plt.ylabel('F stat')
        plt.xlabel('Region Size')
        plt.legend((h[0], h_model[0], h_model_fill),
                   ('null max', 'model', 'model +/- 1 std'))

        return ax

    def plot_f(self, ax=None, size_correct=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            plt.sca(ax)

        ax.set_xscale('log')
        if not size_correct:
            ax.set_yscale('log')
        plt.xlabel('Region Size')

        for idx in range(self.logf_train.shape[0]):
            y = self.logf_train[idx, :]
            if size_correct:
                y = self.get_saf(size=self.size,
                                 f_stats=y,
                                 log_flag=False)
            else:
                # unlog
                y = 10 ** y
            h = plt.scatter(self.size, y)

        if size_correct:
            plt.ylabel('Size Adjusted F Stat')
        else:
            plt.ylabel('F Stat')
        plt.legend((h,), ('null region',))

        return ax

    def plot(self):
        fig, ax = plt.subplots(1, 3)
        self.plot_f(ax=ax[0], size_correct=False)
        self.plot_model(ax=ax[1])
        self.plot_f(ax=ax[2], size_correct=True)


def make_monotonic_up(x, copy=False):
    """ ensures each row is monotonic non decreasing (left to right)

    Args:
        x (np.array): 2 d array
        copy (bool): toggles whether array should be copied

    Returns:
        x (np.array): 2 d array with monotonic rows
    """
    if copy:
        x = np.array(x)

    one_d = False
    if len(x.shape) == 1:
        one_d = True
        x = np.atleast_2d(x)

    x[np.isnan(x)] = -np.inf

    for idx in range(x.shape[1]):
        if idx == 0:
            assert (x[:, 0] != 0).all(), 'size 1 f unobserved'
            continue

        smaller = x[:, idx] < x[:, idx - 1]
        x[smaller, idx] = x[smaller, idx - 1]

    if one_d:
        x = np.squeeze(x, axis=0)

    return x
