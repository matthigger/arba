import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arba.plot import scatter_tree
from arba.region import FeatStat
from arba.seg_graph import SegGraphHistory
from arba.simulate import Model
from arba.space import Mask

obj = 't2'

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
scatter_tree.size_v_t2(sg=sg_hist.tree_history,
                       log_x=True,
                       log_y=True,
                       mask=mask)
