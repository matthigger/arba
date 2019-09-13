import numpy as np

from arba.permute import get_perm_matrix


def test_get_perm_matrix():
    for seed in range(7):
        seed += 1
        perm_matrix = get_perm_matrix(dim=seed, seed=seed)
        assert all(np.sum(perm_matrix, axis=0) == 1), 'col sum not 1'
        assert all(np.sum(perm_matrix, axis=0) == 1), 'row sum not 1'
