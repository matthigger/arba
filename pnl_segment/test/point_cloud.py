import pathlib

import nibabel as nib
import numpy as np

from pnl_segment.space.point_cloud_xyz import PointCloudXYZ

folder_data = pathlib.Path(__file__).parent / 'data'

f_trk = folder_data / 'af.right.trk'
f_ref = folder_data / 'b0.nii.gz'
f_mask = folder_data / 'af.right.mask.nii.gz'

pc_xyz = PointCloudXYZ.from_tract(f_trk)
assert len(pc_xyz) == 17699, 'not all points read in'

pc_ijk = pc_xyz.to_ijk(f_ref)
pc_ijk.discard_doubles()
assert len(pc_ijk) == 2547, 'did not remove redundant points'

# this was used to generate f_mask initially.  f_mask was then visually
# inspected to ensure it covers af.right.trk via trackvis
# pc.to_nii(f_mask)


mask_expected = nib.load(str(f_mask)).get_data()
mask_observed = pc_ijk.to_array()
assert np.array_equal(mask_expected, mask_observed), \
    'mask produced by to_array() doesnt match expected'
