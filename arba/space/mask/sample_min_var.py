import random

import numpy as np

from ..point_cloud import PointCloud
from ..ref_space import get_ref


def sample_mask_min_var(num_vox, ijk_fs_dict, ref=None):
    """ samples a mask by growing a region to minimize variance of features

    Args:
        num_vox (int): total number of voxels
        ijk_fs_dict (dict): keys are tuple of ijk, values are feat_stat
        ref : defines ref space (any input to get_ref())

    Returns:
        mask (Mask): mask of min variance region
    """
    # get ref
    ref = get_ref(ref)
    if ref is None:
        raise AttributeError('ref required, see get_ref()')

    # choose random voxel as init
    min_ijk = random.choice(list(ijk_fs_dict.keys()))
    fs = ijk_fs_dict[min_ijk]

    # init mask (as a PointCloud) and its neighbor set
    pc = PointCloud({}, ref=get_ref(ref))
    ijk_neighbors = set()

    offsets = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
    while True:
        # add new voxel
        assert min_ijk not in pc, 'min_ijk already in mask ...'
        pc.add(min_ijk)

        # check if enough voxels have been added
        if len(pc) >= num_vox:
            break

        # update neighbor list
        ijk_new = {tuple((np.array(min_ijk) + _offset).astype(int))
                   for _offset in offsets}
        ijk_neighbors |= ijk_new
        ijk_neighbors -= pc

        # find neighbor which yields minimal covariance
        min_cov = np.inf
        if not ijk_neighbors:
            raise RuntimeError('empty neighbor set')

        ijk_to_rm = set()
        for ijk in ijk_neighbors:
            try:
                _fs = fs + ijk_fs_dict[ijk]
            except KeyError:
                # voxel not in ijk_fs_dict
                ijk_to_rm.add(ijk)

            if _fs.cov_det < min_cov:
                min_ijk = ijk
                min_cov = _fs.cov_det
        ijk_neighbors -= ijk_to_rm

    assert len(pc) == num_vox, 'invalid mask size sampled'

    # build output mask
    return pc.to_mask()
