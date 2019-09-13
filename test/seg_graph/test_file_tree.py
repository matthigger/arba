import string
from copy import deepcopy

import pytest
from arba.simulate import *


@pytest.fixture
def data_img(mu=np.array((0, 0)), cov=np.eye(2), shape=(3, 5, 7), n=10):
    """ builds files of random noise in 2d, returns data_img

    Args:
        mu (np.array): mean of features
        cov (np.array): covariance of features
        shape (tuple): shape of images
        n (int): number of img in tree
    """

    np.random.seed(1)

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

    yield FileTree(sbj_feat_data_img)

    # delete files
    for sbj, d in sbj_feat_data_img.items():
        for feat, f in d.items():
            os.remove(str(f))


def test_len(data_img):
    assert len(data_img) == 10, 'data_img.__len__ error'


def test_memory(data_img):
    # load data_img, perform a few add operations
    data_img.load()
    x_init = deepcopy(data_img.data)

    # perform a few add operations
    np.random.seed(1)
    mask = np.random.uniform(size=data_img.ref.shape) < .5
    data_img.add(10, mask=mask)
    data_img.add(-10, mask=np.logical_not(mask))

    # save data
    x = data_img.data

    # unload and load data, should re-apply add operations
    data_img.unload()
    data_img.load()

    assert np.allclose(x, data_img.data), 'memory failure'

    data_img.reset_hist()
    data_img.load()

    assert np.allclose(x_init, data_img.data), 'reset failure'
