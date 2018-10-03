import numpy as np


class Effect:
    """ adds an effect to an image

    note: only iid normal effects are supported

    Attributes:
        mean (np.array): average offset of effect on a voxel
        cov (np.array): square matrix, noise power
        mask (np.array): binary array, describes effect location

    >>> np.random.seed(1)
    >>> e = Effect(mean=0, cov=1, mask=np.eye(3))
    >>> e.apply(np.zeros((3, 3)))
    array([[ 1.62434536,  0.        ,  0.        ],
           [ 0.        , -0.61175641,  0.        ],
           [ 0.        ,  0.        , -0.52817175]])
    >>> e = Effect(mean=[0, 0], cov=np.eye(2), mask=np.eye(3))
    >>> e.apply(np.zeros((3, 3, 2)))
    array([[[-1.07296862,  0.86540763],
            [ 0.        ,  0.        ],
            [ 0.        ,  0.        ]],
    <BLANKLINE>
           [[ 0.        ,  0.        ],
            [-2.3015387 ,  1.74481176],
            [ 0.        ,  0.        ]],
    <BLANKLINE>
           [[ 0.        ,  0.        ],
            [ 0.        ,  0.        ],
            [-0.7612069 ,  0.3190391 ]]])
    """

    def __init__(self, mask, mean, cov):
        self._mask = mask > 0
        self.mean = np.atleast_1d(mean)
        self.cov = np.atleast_2d(cov)
        self._len = int(sum(mask.flatten()))

    @property
    def mask(self):
        return self._mask

    def __len__(self):
        return self._len

    def apply(self, x):
        """ applies effect to img x """

        num_dim_img = len(self.mask.shape)
        if not np.array_equal(x.shape[:num_dim_img], self.mask.shape) or \
                len(x.shape) > len(self.mask.shape) + 1:
            raise AttributeError('mask shape and array shape mismatch')

        shape_init = x.shape
        if np.array_equal(shape_init, self.mask.shape):
            # add a feature dimension
            x = np.expand_dims(x, axis=num_dim_img)

        # add noise
        noise = np.vstack(np.random.multivariate_normal(mean=self.mean,
                                                        cov=self.cov,
                                                        size=len(self))).T
        for idx, noise_vec in enumerate(noise):
            x[self.mask, idx] += noise_vec

        if not np.array_equal(shape_init, x.shape):
            # return to original size to be polite
            x = np.reshape(x, shape_init)

        return x
