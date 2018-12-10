import numpy as np

from pnl_segment.adaptive.data import FileTree
from pnl_segment.adaptive.feat_stat import FeatStat
from pnl_segment.adaptive.part_graph_factory import part_graph_factory
from pnl_segment.seg_graph.scatter_tree import size_v_mahalanobis
from pnl_segment.simulate.effect import Effect
from pnl_segment.simulate.mask import Mask
from pnl_segment.point_cloud.ref_space import RefSpace

obj = 'max_maha'
d = 2
mu = np.zeros(d)
cov = np.eye(d)
snr = 1

dim_size = 10
dim_eff = 4

# build mask_effect
ref = RefSpace(shape=tuple([dim_size] * 3))
mask_effect = np.zeros(ref.shape)
mask_effect[:dim_eff, :dim_eff, :dim_eff] = 1
mask_effect = Mask(mask_effect, ref=ref)

# build file tree
ft_dict = dict()
ft_dict['null'] = FileTree()
ft_dict['null'].ref = ref
for ijk, _ in np.ndenumerate(mask_effect.x):
    ft_dict['null'].ijk_fs_dict[ijk] = FeatStat(n=1, mu=mu, cov=cov)

# build file_tree of 'effect'
effect = Effect.from_data(ft_dict['null'].ijk_fs_dict,
                          mask=mask_effect, snr=snr)
ft_dict['effect'] = effect.apply_to_file_tree(ft_dict['null'])

# build part_graph and reduce
pg_hist = part_graph_factory(obj=obj, file_tree_dict=ft_dict,
                             history=True)
pg_hist.reduce_to(1, verbose=True)

# seg_graph
size_v_mahalanobis(pg=pg_hist.tree_history, mask=mask_effect)
