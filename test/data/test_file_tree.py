import string

import pytest

from arba.data import *


@pytest.fixture
def data_img(mu=np.array((0, 0)), cov=None, shape=(3, 5, 7), n=10,
              seed=1):
    """ builds files of random noise in 2d, returns data_img

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
    sbj_feat_data_img = dict()
    for idx in range(n):
        sbj_feat_data_img[idx] = write_nii(idx)

    yield DataImage(sbj_feat_data_img, fnc_list=[scale_normalize])

    # delete files
    for sbj, d in sbj_feat_data_img.items():
        for feat, f in d.items():
            os.remove(str(f))


def test_all(data_img):
    assert len(data_img) == 10, 'data_img.__len__ error'

    mask = Mask(np.ones(data_img.ref.shape), ref=data_img.ref)
    pc = PointCloud.from_mask(mask)

    with data_img.loaded():
        # check space
        assert data_img.mask == mask, 'invalid mask after load'

        # original data should have unequal variance
        assert np.allclose(np.diag(data_img.fs.cov),
                           np.array((1, 2)), rtol=1e-1), 'data not unit scaled'

        # compute feat_stat of data, should be 0 mean, unit variance
        _data = data_img.data[data_img.mask, :, :]
        shape = (len(data_img) * data_img.num_sbj, data_img.num_feat)
        _data = _data.reshape(shape, order='F')
        fs = FeatStat.from_array(_data.T)
        assert np.allclose(fs.mu, 0), 'data not centered'
        assert np.allclose(np.diag(fs.cov), 1), 'data not unit scale'

        f_out = data_img.to_nii(feat='A')
        ref = get_ref(f_out)
        assert ref == data_img.ref, 'ref mismatch'
