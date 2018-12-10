import pytest

from pnl_segment.space import *

n = 4


@pytest.fixture
def point_cloud():
    x_idx_set = {(x, x, x) for x in range(n)}
    ref = RefSpace(affine=np.eye(4), shape=(n, n, n))
    return PointCloud(x_idx_set, ref=ref)


def test_str(point_cloud):
    assert str(point_cloud) == f'PointCloud w/ {n} pts'


def test_swap_ref(point_cloud):
    # build ref_to
    affine = np.eye(4) / 2
    affine[-1, -1] = 1
    ref_to = RefSpace(affine=affine, shape=(n, n))

    pc_swap_ref = point_cloud.swap_ref(ref_to)

    # if affine scale cut in half, values double
    assert set(pc_swap_ref) == {(x * 2, x * 2, x * 2) for x in range(n)}


def test_eq(point_cloud):
    assert point_cloud == point_cloud


def test_to_from_mask(point_cloud):
    mask = point_cloud.to_mask()
    point_cloud_too = PointCloud.from_mask(mask)
    assert point_cloud == point_cloud_too


def test_from_tract():
    folder = pathlib.Path(__file__).parent
    f_trk = folder / 'af.right.trk'
    f_nii = folder / 'af.right.mask.nii.gz'

    # read in 'expected' mask
    mask_expected = Mask.from_nii(f_nii)

    # read in trk, convert to mask
    mask = PointCloud.from_tract(f_trk).to_mask(ref=mask_expected.ref)

    assert mask == mask_expected
