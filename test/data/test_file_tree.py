import string

import pytest

from arba.data import *


@pytest.fixture
def file_tree(mu=np.array((0, 0)), cov=None, shape=(3, 5, 7), n=10,
              seed=1):
    """ builds files of random noise in 2d, returns file_tree

    Args:
        mu (np.array): mean of features
        cov (np.array): covariance of features
        shape (tuple): shape of images
        n (int): number of img in tree
    """

    np.random.seed(seed)
    d = len(mu)
    cov = np.diag(np.array(range(d)) + 1)

    def write_nii(idx):
        """ returns dict of images, one per feature
        """
        # build array
        x = np.random.multivariate_normal(mean=mu, cov=cov, size=shape)

        # get temporary file
        h, f_template = tempfile.mkstemp(suffix=f'_feat_sbj{idx}.nii.gz')
        os.close(h)

        # write output
        f_dict = dict()
        num_feat = len(mu)
        for feat_idx, feat in enumerate(string.ascii_uppercase[:num_feat]):
            f = str(f_template).replace('feat', feat)
            img = nib.Nifti1Image(x[:, :, :, feat_idx], affine=np.eye(4))
            img.to_filename(f)
            f_dict[feat] = f

        return f_dict

    # build file tree
    sbj_feat_file_tree = dict()
    for idx in range(n):
        sbj_feat_file_tree[idx] = write_nii(idx)

    yield DataImage(sbj_feat_file_tree, fnc_list=[scale_normalize])

    # delete files
    for sbj, d in sbj_feat_file_tree.items():
        for feat, f in d.items():
            os.remove(str(f))


def test_all(file_tree):
    assert len(file_tree) == 10, 'file_tree.__len__ error'

    mask = Mask(np.ones(file_tree.ref.shape), ref=file_tree.ref)
    pc = PointCloud.from_mask(mask)

    with file_tree.loaded():
        # check space
        assert file_tree.mask == mask, 'invalid mask after load'

        # original data should have unequal variance
        assert np.allclose(np.diag(file_tree.fs.cov),
                           np.array((1, 2)), rtol=1e-1), 'data not unit scaled'

        # compute feat_stat of data, should be 0 mean, unit variance
        _data = file_tree.data[file_tree.mask, :, :]
        shape = (len(file_tree) * file_tree.num_sbj, file_tree.d)
        _data = _data.reshape(shape, order='F')
        fs = FeatStat.from_array(_data.T)
        assert np.allclose(fs.mu, 0), 'data not centered'
        assert np.allclose(np.diag(fs.cov), 1), 'data not unit scale'

        f_out = file_tree.to_nii(feat='A')
        ref = get_ref(f_out)
        assert ref == file_tree.ref, 'ref mismatch'
