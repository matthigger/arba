import pytest

from pnl_segment.space import *


@pytest.fixture
def mask():
    # __new__()
    x = np.eye(4)
    ref = RefSpace(affine=np.eye(4), shape=(4, 4))
    return Mask(x, ref)


@pytest.fixture
def f_nii(mask):
    # get temp file
    f, f_nii = tempfile.mkstemp(suffix='.nii.gz')
    os.close(f)

    yield f_nii

    # cleanup
    os.remove(f_nii)


def test_eq(mask):
    assert mask == mask, '__eq__()'


def test_dilate(mask):
    dilated_mask = mask.dilate(1)
    dilated_mask_expected = Mask([[1, 1, 0, 0],
                                  [1, 1, 1, 0],
                                  [0, 1, 1, 1],
                                  [0, 0, 1, 1]], ref=dilated_mask.ref)
    assert dilated_mask == dilated_mask_expected, 'dilate()'


def test_len(mask):
    assert len(mask) == 4, '__len__()'


def test_array_finalize(mask):
    assert isinstance(mask + 1, Mask), '__array_finalize__()'


def test_to_nii_and_from_nii_and_from_img(mask, f_nii):
    mask.to_nii(f_nii)
    mask_too = Mask.from_nii(f_nii)

    assert mask_too == mask, 'to_nii(), from_nii() or from_img()'
