import numpy as np

from ..point_cloud import PointCloud


def sample_mask_min_var(num_vox, data_img, prior_array=None,
                        cov_measure=np.trace):
    """ samples a mask by growing a region to minimize variance of features

    Args:
        num_vox (int): total number of voxels
        data_img (DataImage):
        cov_measure (fnc): how should cov be measured? either np.linalg.det or
                           np.trace

    Returns:
        mask (Mask): mask of min variance region
    """

    # choose random voxel as init
    if prior_array is None:
        prior_array = data_img.mask
    p = np.array(prior_array / prior_array.sum())

    # choose random starting voxel (ijk_min_fs)
    idx = np.random.choice(range(p.size), size=1, p=p.flatten())
    ijk_min_fs = np.unravel_index(idx[0], prior_array.shape)
    fs = data_img.get_fs(ijk=ijk_min_fs)

    # init mask (as a PointCloud) and its neighbor set
    pc = PointCloud({}, ref=data_img.ref)

    ijk_neighbors = set()
    offsets = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
    while True:
        # add voxel
        assert ijk_min_fs not in pc, 'ijk_min_fs already in mask ...'
        pc.add(ijk_min_fs)

        # check if enough voxels have been added
        if len(pc) >= num_vox:
            break

        # update neighbor list
        ijk_new = {tuple((np.array(ijk_min_fs) + _offset).astype(int))
                   for _offset in offsets}
        ijk_new = set(filter(lambda ijk: prior_array[ijk], ijk_new))
        ijk_neighbors |= ijk_new
        ijk_neighbors -= pc

        if not ijk_neighbors:
            raise RuntimeError('empty neighbor set')

        # find neighbor which yields minimal covariance
        min_cov = np.inf
        for ijk in ijk_neighbors:
            _fs = fs + data_img.get_fs(ijk=ijk)

            _cov = cov_measure(_fs.cov)
            if _cov < min_cov:
                ijk_min_fs = ijk
                min_cov = _cov

    assert len(pc) == num_vox, 'invalid mask size sampled'

    # build output mask
    return pc.to_mask()
