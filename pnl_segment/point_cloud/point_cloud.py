import numpy as np

from . import ref_space


class PointCloud:
    """ super class for PointCloudIJK and PointCloudXYZ

    >>> x =np.vstack((np.eye(2), np.eye(2)))
    >>> print(x)
    [[1. 0.]
     [0. 1.]
     [1. 0.]
     [0. 1.]]
    >>> pc = PointCloud(x)
    >>> print(pc)
    PointCloud with 4 pts
    >>> len(pc)
    4
    >>> pc.dim
    2
    >>> pc.center
    array([0.5, 0.5])
    >>> pc.discard_doubles()
    >>> print(pc)
    PointCloud with 2 pts
    """

    def __init__(self, x, ref):
        if len(x.shape) == 1:
            raise AttributeError('must init with 2d array')

        self.x = np.array(x)
        self.ref = ref_space.get_ref(ref)

    def swap_ref(self, ref):
        """ swaps orientation to a new ref
        """
        x = self.ref.to_rasmm(self.x)

        ref_to = ref_space.get_ref(ref)
        x = ref_to.from_rasmm(x)

        return self.__class__(x, ref=ref)

    def __len__(self):
        return self.x.shape[0]

    @property
    def dim(self):
        return self.x.shape[1]

    @property
    def center(self):
        return np.mean(self.x, axis=0)

    def __str__(self):
        return f'{type(self).__name__} with {len(self)} pts'

    def discard_doubles(self):
        self.x = np.unique(self.x, axis=0)

    def __eq__(self, other):
        # note: resorting pts spoils equality
        return np.array_equal(self.x, other.x)

    def __add__(self, other):
        """ adds points to the point cloud
        """
        if type(self) != type(other):
            TypeError(f'type mismatch: {type(self)}, {type(other)}')

        if not isinstance(other, PointCloud) and other == 0:
            # allows sum()
            return self.__class__(self.x, ref=self.ref)

        # ensure affines are equivilent
        if self.ref != other.ref:
            raise AttributeError('cant merge different spaces')

        return self.__class__(x=np.vstack((self.x, other.x)), ref=self.ref)

    __radd__ = __add__
