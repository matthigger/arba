import pytest

from arba.simulate import *


@pytest.fixture
def ft_tuple(var=1, eff=10, shape=(10, 10, 10), shape_eff=(4, 5, 6), n_null=10,
             n_effect=11):
    """ builds two file trees with some effect

    Args:
        var (float): variance of white gaussian noise
        eff (float): magnitude of effect
        shape (tuple): shape of images
        shape_eff (tuple): size of effect, note its in the first few vox in
                           each dimension
        n_null (int): number of images without an effect
        n_effect (int): number of images with the effect
    """

    np.random.seed(1)

    eff_slice = np.s_[0:shape_eff[0], 0:shape_eff[1], 0:shape_eff[2]]

    def write_nii(eff, label=''):
        # build array
        x = np.random.normal(scale=np.sqrt(var), size=shape)
        x[eff_slice] += eff

        # get tempfile
        h, f = tempfile.mkstemp(suffix=f'{label}.nii.gz')
        os.close(h)

        # write output
        img = nib.Nifti1Image(x, affine=np.eye(4))
        img.to_filename(f)

        return f

    def get_data_img(eff, n, label=''):
        sbj_feat_data_img = defaultdict(dict)
        for idx in range(n):
            sbj = f'{label}_{idx}'
            sbj_feat_data_img[sbj]['feat'] = write_nii(eff, label=sbj)
        return FileTree(sbj_feat_data_img)

    ft0 = get_data_img(eff=0, n=n_null, label='null')
    ft1 = get_data_img(eff=eff, n=n_effect, label='effect')

    return ft0, ft1
