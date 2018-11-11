from pnl_segment.space import *
import pathlib

f_mask_nii = pathlib.Path(__file__).parents[1] / 'data' / 'mask.nii.gz'

m = Mask.from_nii(f_mask_nii)
len(m)
pc = m.to_point_cloud()