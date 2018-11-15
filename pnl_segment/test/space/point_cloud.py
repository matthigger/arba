from pnl_segment.space import *

folder_data = pathlib.Path(__file__).parents[1] / 'data'
f_trk = folder_data / 'af.right.trk'
pc = PointCloud.from_tract(f_trk)

f_fa = folder_data / 'FA.nii.gz'
ref = get_ref(f_fa)
f_trk_nii = folder_data / 'af.right.mask.nii.gz'
pc.to_mask(ref).to_nii(f_trk_nii)

# build dummy ref space
shape = (3, 3)
affine = np.eye(3) * 2
affine[-1, -1] = 1
ref = RefSpace(affine, shape=shape)

#
pc = PointCloud.from_mask(np.eye(2), ref=ref)
pc_rasmm = pc.swap_ref(round=True)
pc_union = pc - pc
