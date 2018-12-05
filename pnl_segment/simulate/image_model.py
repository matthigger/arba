from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pnl_segment.seg_graph import FeatStat


class ImageModel:
    def __init__(self, shape, ijk_fs_dict):
        self.shape = shape
        self.ijk_fs_dict = ijk_fs_dict

    @staticmethod
    def from_images(array_list):
        ijk_sample_dict = defaultdict(list)
        shape = next(iter(array_list)).shape

        # aggregate per voxel
        for x in array_list:
            if x.shape != shape:
                raise AttributeError('mismatched shape')

            for ijk, _x in np.ndenumerate(x):
                ijk_sample_dict[ijk].append(_x)

        # estimate normal
        ijk_fs_dict = {ijk: FeatStat.from_iter(x_list)
                       for ijk, x_list in ijk_sample_dict.items()}

        return ImageModel(shape, ijk_fs_dict)

    def _combine(self, ijk_list):
        fs = sum(self.ijk_fs_dict[ijk] for ijk in ijk_list)

        for ijk in ijk_list:
            self.ijk_fs_dict[ijk] = fs

    def combine_by_seg(self, seg):

        for val in np.unique(seg.flatten()):
            ijk_list = np.vstack(np.where(seg == val)).T
            ijk_list = [tuple(x) for x in ijk_list]
            self._combine(ijk_list)

    def sample(self):
        x = np.ones(self.shape) * np.nan
        for ijk, fs in self.ijk_fs_dict.items():
            x[ijk] = fs.to_normal().rvs()
        return x


if __name__ == '__main__':
    import pathlib

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
    model_list = [ImageModel(shape, ijk_fs_dict)
                  for ijk_fs_dict in ijk_fs_dict_list]

    # sample images
    np.random.seed(1)
    sample_list_dict = {'truth': [[im.sample() for _ in range(n_sample)]
                                  for im in model_list]}

    # build estimated models
    seg_im_list_dict = dict()
    for seg_label, seg in seg_dict.items():
        im_list = [ImageModel.from_images(array_list)
                   for array_list in sample_list_dict['truth']]
        for im in im_list:
            im.combine_by_seg(seg)

        seg_im_list_dict[seg_label] = im_list

    # sample images
    for label, im_list in seg_im_list_dict.items():
        sample_list_dict[label] = [[im.sample() for _ in range(n_sample)]
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
