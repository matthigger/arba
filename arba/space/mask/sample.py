import random
from collections import Counter
from copy import copy

import numpy as np
from scipy.ndimage import binary_dilation
from skimage import measure

from .mask import Mask
from ..point_cloud import PointCloud
from ..ref_space import get_ref


def sample_mask(prior_array, radius=None, seg_array=None, ref=None,
                num_vox=None):
    """n effect centers chosen and dilated to build mask of affected volume

    Args:
        prior_array (np.array): has same shape as image.  values are
                                unweighted prob of effect center.
                                note that prior array also masks effect (no
                                effect exists where prior_array <= 0)
        radius (int, tuple): if int, acts as effect radius: center is
                             dilated this many times.  if tuple, expected
                             as shape of effect
        seg_array (np.array): segmentation array, if passed constrains
                              effect to the region of its center. has
                              same shape as prior_array
        ref : reference space of output mask
        num_vox (int): if passed, will ensure mask has exactly total_vox
                         points

    Returns:
        eff_mask (np.array): array of 1s where effect is present
    """
    # check that inputs are consistent
    assert (radius is None) != (num_vox is None), \
        'radius xor num_vox required'

    # if num_vox passed, there are constraints on prior_array and seg_array
    if num_vox is not None:
        if seg_array is not None:
            # avoid memory intersection with prior_array passed
            prior_array = copy(prior_array)

            # selection must be in intersection of seg_array + prior_array
            prior_array[np.logical_not(seg_array)] = 0
            seg_array[np.logical_not(prior_array)] = 0

            # ensure each idx of seg_array is connected
            seg_array = measure.label(seg_array)

            # rm idx in seg_array which don't have at least num_vox voxels
            c = Counter(seg_array.flatten())
            idx_invalid = [idx for idx, count in c.items()
                           if count < num_vox]
            assert len(idx_invalid) < len(c), \
                f'seg_array doesnt have regions with {num_vox} voxels'
            for idx in idx_invalid:
                prior_array[seg_array == idx] = 0

        assert prior_array.astype(bool).sum() >= num_vox, \
            f'prior_array doesnt have {num_vox} valid voxels'

    # choose effect center via prior array
    p = np.array(prior_array / prior_array.sum())
    idx = np.random.choice(range(p.size), size=1, p=p.flatten())
    center_ijk = np.unravel_index(idx[0], prior_array.shape)

    # init mask
    mask = np.zeros(prior_array.shape)
    mask[center_ijk] = 1

    # meta_mask is a mask of the mask (contains all the possible voxels
    # that an effect mask could contain)
    meta_mask = prior_array.astype(bool)
    if seg_array is not None:
        meta_mask = np.logical_and(meta_mask,
                                   seg_array == seg_array[center_ijk])

    # perform dilation
    if radius is not None:
        if isinstance(radius, int):
            # dilate radius number of times
            mask = binary_dilation(mask, iterations=radius, mask=meta_mask)
        elif len(radius) == 3:
            # shape specified
            for delta in np.ndindex(radius):
                mask[tuple(center_ijk + delta)] = 1
            mask = np.logical_and(mask, meta_mask)
        else:
            raise AttributeError('invalid radius, must be int or tuple')
        return Mask(mask, ref=get_ref(ref))

    # dilate mask until it has > num_vox voxels
    assert num_vox > 1, 'num_vox must be > 1'
    while mask.sum() < num_vox:
        mask_last = mask
        mask = binary_dilation(mask, iterations=1, mask=meta_mask,
                               structure=np.ones((3, 3, 3)))

    # rm some of the voxels added in last dilation until mask has num_vox
    shell_mask = Mask(np.logical_and(mask, np.logical_not(mask_last)))
    shell_pc = PointCloud.from_mask(shell_mask)
    ijk_to_rm = random.sample(shell_pc, k=mask.sum() - num_vox)
    for ijk in ijk_to_rm:
        mask[ijk] = False

    assert mask.sum() == num_vox, 'mask doesnt have num_vox voxels'

    return Mask(mask, ref=get_ref(ref))
