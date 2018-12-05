import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pnl_segment.seg_graph import FeatStat
from pnl_segment.simulate import Model

shape = (4, 4)
low = FeatStat(n=1000, mu=-2, cov=1)
mid = FeatStat(n=1000, mu=0, cov=1)
high = FeatStat(n=1000, mu=2, cov=1)
n_sample = 4
cm = plt.get_cmap('viridis')
folder = pathlib.Path('/home/matt/Downloads')

bad_seg = np.array([[1] * 8 + [0] * 8]).reshape(shape)
good_seg = bad_seg.T
seg_dict = {'good': good_seg,
            'bad': bad_seg}

# build ground truth image model
ijk_fs_dict_list = (dict(), dict())
for ijk in np.ndindex(shape):
    if good_seg[ijk]:
        ijk_fs_dict_list[0][ijk] = low
        ijk_fs_dict_list[1][ijk] = high
    else:
        ijk_fs_dict_list[0][ijk] = mid
        ijk_fs_dict_list[1][ijk] = mid
model_list = [Model(ijk_fs_dict, shape=shape)
              for ijk_fs_dict in ijk_fs_dict_list]

# sample images
np.random.seed(1)
sample_list_dict = {'truth': [[im.sample_array() for _ in range(n_sample)]
                              for im in model_list]}

# build estimated models
seg_im_list_dict = dict()
for seg_label, seg in seg_dict.items():
    im_list = [Model.from_arrays(array_list)
               for array_list in sample_list_dict['truth']]
    for im in im_list:
        im.combine_by_seg(seg)

    seg_im_list_dict[seg_label] = im_list

# sample images
for label, im_list in seg_im_list_dict.items():
    sample_list_dict[label] = [[im.sample_array() for _ in range(n_sample)]
                               for im in im_list]

#
for label, sample_list in sample_list_dict.items():
    for grp_idx, l in enumerate(sample_list):
        fig, ax = plt.subplots(1, n_sample)
        for x, _ax in zip(l, ax):
            cax = _ax.imshow(x, cmap=cm, vmin=-3, vmax=3, origin='lower')
            _ax.get_xaxis().set_visible(False)
            _ax.get_yaxis().set_visible(False)
        # import seaborn as sns
        # sns.set(font_scale=1.2)
        # plt.colorbar(cax, ticks=[-2, 0, 2])

        f_out = folder / f'{label}_grp_{grp_idx}.pdf'
        with PdfPages(f_out) as pdf:
            plt.gcf().set_size_inches(8, 2)
            pdf.savefig(plt.gcf(), bbox_inches='tight', pad_inches=0)
            plt.close()
