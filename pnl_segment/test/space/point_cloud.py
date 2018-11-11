from pnl_segment.space import *

# build dummy ref space
shape = (3, 3)
affine = np.eye(3) * 2
affine[-1, -1] = 1
ref = RefSpace(affine, shape=shape)

#
pc = PointCloud.from_mask_array(np.eye(2), ref=ref)
pc_rasmm = pc.swap_ref(round=True)
pc_union = pc - pc