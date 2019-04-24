from collections import defaultdict

import networkx as nx
import nibabel as nib
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..region import RegionWardGrp
from ..space import PointCloud


class SegGraph(nx.Graph):
    """ segmentation graph, stores and merges region objects as a graph

    each region object describes the statistics of populations within some area
    (see RegionWardT2)

    Attributes:
        file_tree (FileTree): file tree
        split (Split):
    """

    def __init__(self, file_tree, split, _add_nodes=True, **kwargs):
        """

        Args:
            file_tree (FileTree): file tree
            split (Split):
            _add_nodes (bool): toggles whether nodes are added, useful
                               internally if empty SegGraph needed
        """
        super().__init__()
        self.file_tree = file_tree
        self.split = split

        if _add_nodes:
            with file_tree.loaded():
                self._add_nodes()
                self.connect_neighbors(**kwargs)

    def _add_nodes(self):
        assert self.file_tree.pc, 'no active area found in file_tree'

        for ijk in self.file_tree.pc:
            # space region occupies
            pc_ijk = PointCloud({tuple(ijk)}, ref=self.file_tree.ref)

            # build and store in graph
            reg = RegionWardGrp.from_data(pc_ijk=pc_ijk,
                                          file_tree=self.file_tree,
                                          split=self.split)
            self.add_node(reg)

    def connect_neighbors(self, edge_directions=np.eye(3), **kwargs):
        """ adds edge between each neighboring region """
        # build ijk_reg_map, maps ijk to a corresponding region
        ijk_reg_map = dict()
        for reg in self.nodes:
            for ijk in reg.pc_ijk:
                ijk_reg_map[ijk] = reg

        # iterate through edge_directions of each ijk to find neighbors
        for ijk0, reg0 in ijk_reg_map.items():
            for offset in edge_directions:
                ijk1 = tuple((ijk0 + offset).astype(int))

                # add edge if ijk1 neighbor corresponds to a region
                if ijk1 in ijk_reg_map.keys():
                    self.add_edge(reg0, ijk_reg_map[ijk1])

    def to_nii(self, f_out, **kwargs):
        """ saves nifti image """
        # build array
        x = self.to_array(shape=self.file_tree.ref.shape, **kwargs)

        # save
        img_out = nib.Nifti1Image(x, self.file_tree.ref.affine)
        img_out.to_filename(str(f_out))

        return img_out

    def to_array(self, fnc=None, fnc_include=None, background=0):
        """ constructs array of region idx """
        if fnc is None:
            if fnc_include is not None:
                nodes = [reg for reg in self.nodes if fnc_include(reg)]
            else:
                nodes = self.nodes
            reg_to_idx = {reg: idx for idx, reg in enumerate(nodes)}

            def fnc(reg):
                return reg_to_idx[reg]

        if fnc_include is None:
            reg_list = list(self.nodes)
        else:
            reg_list = list(filter(fnc_include, self.nodes))

        shape = self.file_tree.ref.shape

        assert (not set().intersection(*[r.pc_ijk for r in self])), \
            'non disjoint regions found'

        # build output array
        x = np.zeros(shape) * background
        for reg in reg_list:
            val = fnc(reg)
            for ijk in reg.pc_ijk:
                x[tuple(ijk)] += val

        return x

    def merge(self, reg_tuple):
        """ combines neighboring regions

        Args:
            reg_tuple (iter): iterator of regions to be combined
        """

        # create rew region
        reg_iter = iter(reg_tuple)
        reg_sum = next(reg_iter)
        for reg in reg_iter:
            reg_sum += reg

        # get neighbor list (before removing any nodes)
        neighbr_list = [self.neighbors(r) for r in reg_tuple]
        neighbr_set = frozenset().union(*neighbr_list)
        neighbr_set -= set(reg_tuple)

        # rm old regions from seg_graph
        for r in reg_tuple:
            self.remove_node(r)

        # add new region to seg_graph
        self.add_node(reg_sum)

        # add edges to all of combined nodes
        for r_neighbr in neighbr_set:
            self.add_edge(reg_sum, r_neighbr)

        return reg_sum

    def merge_by_atlas(self, f_region, verbose=False):
        """ builds atlas region by merging regions per voxel

        Note: regions must all be single voxel at outset
        """

        # find ijk_reg_dict, keys are ijk (tuple), values are reg they are in
        ijk_reg_dict = dict()
        for reg in self.nodes:
            if len(reg.pc_ijk) != 1:
                raise AttributeError('regions must consist of single vox')
            ijk = next(iter(reg.pc_ijk))
            ijk_reg_dict[ijk] = reg

        # load f_region
        img_reg = nib.load(str(f_region))
        x = img_reg.get_data()

        # build reg_idx_reg_dict, keys are reg_idx (via f_region), values are
        # list of reg object which belong to reg_idx
        reg_idx_list = np.unique(x.flatten())
        reg_idx_reg_dict = defaultdict(list)
        ijk_missing = list()
        for reg_idx in reg_idx_list:
            if reg_idx == 0:
                continue
            mask = x == reg_idx
            ijk_array = np.vstack(np.where(mask)).T
            for ijk in ijk_array:
                try:
                    reg = ijk_reg_dict[tuple(ijk)]
                    reg_idx_reg_dict[reg_idx].append(reg)
                except KeyError:
                    ijk_missing.append(tuple(ijk))

        if verbose:
            print(f'missing {len(ijk_missing)} regions (voxels)?')

        # merge
        reg_new_dict = dict()
        for reg_idx, reg_list in tqdm(reg_idx_reg_dict.items(),
                                      desc='combining',
                                      disable=not verbose):
            reg_new_dict[reg_idx] = self.merge(reg_list)

    def get_sig(self, alpha=.05, method='holm', _reg_pval_dict=None):
        """ returns a SegGraph containing only significant regions in self

        Args:
            alpha (float): significance level
            method (str): see multipletests, defaults to Holm's
            _reg_pval_dict (dict):

        Returns:
            sg (SegGraph): contains only significant regions
        """

        # init seg graph
        sg = SegGraph(file_tree=self.file_tree, split=self.split,
                      _add_nodes=False)

        # if self is empty, return empty SegGraph
        if not self.nodes:
            return sg

        if _reg_pval_dict is None:
            _reg_pval_dict = {reg: reg.pval for reg in self.nodes}

        # get only the significant regions
        sig_vec = multipletests(list(_reg_pval_dict.values()),
                                alpha=alpha, method=method)[0]
        reg_sig_list = [r for r, sig in zip(_reg_pval_dict.keys(), sig_vec)
                        if sig]

        # add sig regions
        sg.add_nodes_from(reg_sig_list)

        return sg
