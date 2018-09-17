import pathlib
import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np
from pnl_segment.adaptive import part_graph_factory

# compares n_healthy to n_effect images.  all have gaussian noise added, effect
#  has eff_size added (FA and MD) @ eff_center within eff_radius (vox)
n_healthy = 10
n_effect = 10
noise_power = .01
eff_size = .4
eff_center = (53, 63, 36)
eff_radius = 5

# load files
folder = pathlib.Path(__file__).parent
f_fa = str(folder / 'FA.nii.gz')
f_md = str(folder / 'MD.nii.gz')
f_mask = str(folder / 'mask.nii.gz')

# repeatably random
np.random.seed(1)

# # strip all but data and affine (its already anonymized, but can't hurt to be
# # minimal)
# for f in (f_fa, f_md, f_mask):
#     img = nib.load(f)
#     img_copy = nib.Nifti1Image(img.get_data(), img.affine)
#     img_copy.to_filename(f)

# numpy takes std_dev...
scale = np.sqrt(noise_power)

# build slice obj of effected area
eff_slice = tuple(slice(c - eff_radius, c + eff_radius) for c in eff_center)
eff_shape = tuple([eff_radius * 2] * 3)


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


# build f_img_dict (see part_graph_factory._build_part_graph for ex)
f_img_dict = defaultdict(list)
for _ in range(n_healthy):
    f_img_dict['healthy'].append((sample_img(f_fa),
                                  sample_img(f_md)))
for _ in range(n_effect):
    f_img_dict['effect'].append((sample_img(f_fa, eff_size=eff_size),
                                 sample_img(f_md, eff_size=eff_size)))

pg = part_graph_factory.max_kl(f_img_dict=f_img_dict, verbose=True,
                               f_mask=f_mask)
