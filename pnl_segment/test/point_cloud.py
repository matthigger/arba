import pathlib

import nibabel as nib
import numpy as np

import pnl_segment

f_trk = pathlib.Path(__file__).with_name('af.right.trk')
f_ref = pathlib.Path(__file__).with_name('b0.nii.gz')
f_mask = pathlib.Path(__file__).with_name('af.right.mask.nii.gz')

pc = pnl_segment.PointCloud.from_tract(f_trk, f_ref)
assert len(pc) == 17699, 'not all points read in'

pc.make_ijk_rep()
assert len(pc) == 2547, 'make_ijk_rep() did not remove redundant points'

# this was used to generate f_mask initially.  f_mask was then visually
# inspected to ensure it covers af.right.trk via trackvis
# pc.to_nii(f_mask)


mask_expected = nib.load(str(f_mask)).get_data()
mask_observed = pc.to_array()
assert np.array_equal(mask_expected, mask_observed), \
    'mask produced by to_array() doesnt match expected'
