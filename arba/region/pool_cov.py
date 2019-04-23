class PoolCov:

    @staticmethod
    def from_feat_stat(fs):
        return PoolCov(n=fs.n, value=fs.cov)

    def __init__(self, n, value):
        self.n = n
        self.value = value

    def __add__(self, other):
        return PoolCov.sum((self, other))

    @staticmethod
    def sum(pool_cov_list):
        n = sum(pc.n for pc in pool_cov_list)
        value = 0
        for pc in pool_cov_list:
            value += pc.value * pc.n / n

        return PoolCov(n, value)

    def __str__(self):
        return f'PoolCov({self.n}, {self.value})'


if __name__ == '__main__':
    import numpy as np

    pc1 = PoolCov(1, np.eye(2))
    pc0 = PoolCov(9, np.zeros((2, 2)))
    print(pc0 + pc1)
