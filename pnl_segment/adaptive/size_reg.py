import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


class SizeReg:
    """ size regressor (or size regularizer)
    """

    def __init__(self, log_size=True):
        self.line = LinearRegression()
        self.log_size = log_size

    def fit(self, size, x):
        if self.log_size:
            size = np.log10(size)

        size = np.atleast_2d(size).reshape((-1, 1))
        x = np.atleast_2d(x).reshape((-1, 1))
        self.line.fit(size, x)

    def predict(self, size):
        if self.log_size:
            size = np.log10(size)
        size = np.atleast_2d(size).reshape((-1, 1))
        return self.line.predict(size)

    def plot(self, size, x, scatter=True, mean=True):
        if self.log_size:
            size = np.log10(size)

        x_domain = np.atleast_2d([min(size), max(size)]).T

        sns.set()
        if scatter:
            plt.scatter(size, x, alpha=.3, color='b')

        if mean:
            y_mean = self.line.predict(x_domain)
            plt.plot(x_domain, y_mean, color='r', label='mean')
            plt.legend()
