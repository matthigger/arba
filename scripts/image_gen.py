import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pnl_segment.plot import scatter_tree
from pnl_segment.region import FeatStat
from pnl_segment.seg_graph import SegGraphHistory
from pnl_segment.simulate import Model
from pnl_segment.space import Mask

obj = 'maha'

# taken from "2018_P_INRIA.odp" example
shape = (4, 4, 1)
n = 100
color_fs_dict = {'green': FeatStat(n=10, mu=[0], cov=[1]),
                 'orange': FeatStat(n=10, mu=[-1], cov=[1]),
                 'blue': FeatStat(n=10, mu=[1], cov=[1])}
mask = Mask(np.zeros(shape))

ft_dict = dict()
for grp in ['healthy', 'effect']:
    ijk_fs_dict = dict()
    for ijk in np.ndindex(shape):
        if ijk[0] < 2:
            if grp == 'healthy':
                ijk_fs_dict[ijk] = color_fs_dict['orange']
            else:
                ijk_fs_dict[ijk] = color_fs_dict['blue']
            mask[ijk] = 1
        else:
            ijk_fs_dict[ijk] = color_fs_dict['green']

    image_gen = Model(ijk_fs_dict, shape=shape)
    ft_dict[grp] = image_gen.to_file_tree(n=n)

sg_hist = SegGraphHistory(obj=obj, file_tree_dict=ft_dict)
sg_hist.reduce_to(1)

sns.set(font_scale=1.2)
fig, ax = plt.subplots(1, 1)
scatter_tree.size_v_mahalanobis(sg=sg_hist.tree_history,
                                log_x=True,
                                log_y=True,
                                mask=mask)
