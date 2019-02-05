from pnl_brain.tools import register, transform
from pnl_data.set.cidar_post import folder

folder_fs = folder / 'fs' / '01193'
f_t1 = folder_fs / 'T1.mgz'
f_fa = folder / 'dti' / '01193_FA.nii.gz'
f_t1_in_dti = str(f_t1).replace('.mgz', '_in_dti.nii.gz')

d = {'f_fix': f_fa,
     'f_move': f_t1,
     'folder_out': folder_fs,
     'f_warped': f_t1_in_dti,
     'f_mask_fix': f_fa,
     'f_mask_move': f_t1}

reg_files = register.reg_affine_syn_ants_def(d)
f_affine = folder_fs / '011930GenericAffine.mat'
f_warp = folder_fs / '011931Warp.nii.gz'

trans_iter = [(f_affine, 0), (f_warp, 0)]

for f in ['aparc.a2009s+aseg.mgz', 'aparc+aseg.mgz', 'aseg.mgz']:
    f = folder_fs / f
    f_out = str(f).replace('.mgz', '_in_dti.nii.gz')
    transform.map_affine_warp(f_in=f, f_out=f_out, f_ref=f_fa,
                              interp='NearestNeighbor', trans_iter=trans_iter)
