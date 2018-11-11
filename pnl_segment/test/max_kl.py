import pathlib
import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np
from mh_pytools import file
from pnl_segment.seg_graph import factory

# compares n_healthy to n_effect images.  all have gaussian noise added, effect
#  has eff_size added (FA and MD) @ eff_center within eff_radius (vox)
n_healthy = 20
n_effect = 20
noise_power = .01
eff_size = .05
eff_center = (53, 63, 36)
eff_radius = 2
mask_radius = 6

# files
folder = pathlib.Path(__file__).parent
folder_data = folder / 'data'
f_fa = str(folder_data / 'FA.nii.gz')
f_md = str(folder_data / 'MD.nii.gz')
f_mask = str(folder_data / 'mask.nii.gz')
img_label = ['FA', 'MD']

# repeatably random
np.random.seed(1)


# # strip all but data and affine (its already anonymized, but can't hurt to be
# # minimal)
# for f in (f_fa, f_md, f_mask):
#     img = nib.load(f)
#     img_copy = nib.Nifti1Image(img.get_data(), img.affine)
#     img_copy.to_filename(f)


def build_mask(r, f=None):
    # load
    img_mask = nib.load(f_mask)

    # build mask
    x = np.zeros(img_mask.shape)
    area = tuple(slice(c - r, c + r) for c in eff_center)
    x[area] = 1
    x = np.logical_and(x, img_mask.get_data())

    if f is None:
        _, f = tempfile.mkstemp(suffix='.nii.gz')
    img_mask = nib.Nifti1Image(x.astype(float), img_mask.affine)
    img_mask.to_filename(str(f))
    return f


# compute values needed in sample_img
eff_slice = tuple(slice(c - eff_radius, c + eff_radius) for c in eff_center)
eff_shape = tuple([eff_radius * 2] * 3)
scale = np.sqrt(noise_power)


def sample_img(f, eff_size=0):
    """ 'samples' an image whose ground truth is f

    note: this adds noise outside of mask, we'll mask it in the algorithm
    """

    # read in data
    img = nib.load(f)
    x = img.get_data()

    # add noise
    x += np.random.normal(0, scale=scale, size=img.shape)

    if eff_size:
        x[eff_slice] += np.ones(shape=eff_shape) * eff_size

    # save to file
    _, f = tempfile.mkstemp(suffix='.nii.gz')
    img_out = nib.Nifti1Image(x, img.affine)
    img_out.to_filename(f)

    return f


# build masks (quicker computation)
f_mask = build_mask(mask_radius)
f_mask_effect = folder_data / 'mask_effect.nii.gz'
build_mask(eff_radius, f=f_mask_effect)

# build f_img_dict (see seg_graph_factory._build_part_graph for ex)
f_img_dict = defaultdict(list)
for _ in range(n_healthy):
    f_img_dict['healthy'].append((sample_img(f_fa),
                                  sample_img(f_md)))
for _ in range(n_effect):
    f_img_dict['effect'].append((sample_img(f_fa, eff_size=eff_size),
                                 sample_img(f_md, eff_size=eff_size)))

pg = factory.max_kl(f_img_dict=f_img_dict, verbose=True,
                    f_mask=f_mask, history=True, img_label=img_label)

pg.reduce_to(1, edge_per_step=1e-6)
f_part_graph = folder_data / 'seg_graph.p.gz'
file.save(pg, f_part_graph)
