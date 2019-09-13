import numpy as np


def get_perm_matrix(dim, seed=None):
    """ builds permutation matrix

    Args:
        dim (int): dimension of permutation matrix
        seed: random seed

    Returns:
        perm_matrix (np.array): (dim, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    perm_matrix = np.zeros((dim, dim))
    for idx_from, idx_to in enumerate(np.random.permutation(range(dim))):
        perm_matrix[idx_from, idx_to] = 1

    return perm_matrix
