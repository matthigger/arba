import pathlib
from collections import defaultdict

import nibabel as nib
from tqdm import tqdm

from pnl_data.set import intrust
from arba.seg_graph import FeatStat, FeatStatEmpty
from arba.space import Mask, PointCloud
from pnl_brain.region import FSRegion

# load data
folder = pathlib.Path(
    '/home/matt/dropbox_pnl/data/intrust (Selective Sync Conflict)/vote_parc/015_NAA_001')
f_fa_glob = '**/*_warped.nii.gz'
f_fs_glob = '**/*fsindwi.nii.gz'

sbj_f_fa_dict = dict()
for f in folder.glob(f_fa_glob):
    s = str(f).split('/')[8]
    name = intrust.get_name(s)
    sbj_f_fa_dict[name] = f

sbj_f_fs_dict = dict()
for f in folder.glob(f_fs_glob):
    s = str(f).split('/')[8]
    name = intrust.get_name(s)
    sbj_f_fs_dict[name] = f

# build mask
mask_iter = (Mask.from_nii(f) for f in sbj_f_fs_dict.values())
mask = sum(mask_iter) > 0
mask_pc = PointCloud.from_mask(mask)


# build feat stat per class per voxel
def fnc():
    return defaultdict(FeatStatEmpty)


ijk_grp_fs_tree = defaultdict(fnc)

for sbj, f_fa in tqdm(sbj_f_fa_dict.items(), desc='load per sbj'):
    f_fs = sbj_f_fs_dict[sbj]
    fa = nib.load(str(f_fa)).get_data()
    fs = nib.load(str(f_fs)).get_data()
    for ijk in mask_pc:
        raise NotImplementedError('must normalize fa')
        _fa = fa[ijk]
        grp = fs_reg = FSRegion(fs[ijk])

        ijk_grp_fs_tree[ijk][grp] += FeatStat(n=1, mu=_fa, cov=0)
