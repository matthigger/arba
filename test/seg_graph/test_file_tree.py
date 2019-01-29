import string
from copy import deepcopy

import pytest
from pnl_segment.simulate import *


@pytest.fixture
def file_tree(mu=np.array((0, 0)), cov=np.eye(2), shape=(3, 5, 7), n=10):
    """ builds files of random noise in 2d, returns file_tree

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
    sbj_feat_file_tree = dict()
    for idx in range(n):
        sbj_feat_file_tree[idx] = write_nii(idx)

    yield FileTree(sbj_feat_file_tree)

    # delete files
    for sbj, d in sbj_feat_file_tree.items():
        for feat, f in d.items():
            os.remove(str(f))


def test_len(file_tree):
    assert len(file_tree) == 10, 'file_tree.__len__ error'


def test_memory(file_tree):
    # load file_tree, perform a few add operations
    file_tree.load()
    x_init = deepcopy(file_tree.data)

    # perform a few add operations
    np.random.seed(1)
    mask = np.random.uniform(size=file_tree.ref.shape) < .5
    file_tree.add(10, mask=mask)
    file_tree.add(-10, mask=np.logical_not(mask))

    # save data
    x = file_tree.data

    # unload and load data, should re-apply add operations
    file_tree.unload()
    file_tree.load()

    assert np.allclose(x, file_tree.data), 'memory failure'

    file_tree.reset()
    file_tree.load()

    assert np.allclose(x_init, file_tree.data), 'reset failure'
