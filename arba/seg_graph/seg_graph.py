from collections import defaultdict

import networkx as nx
import nibabel as nib
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..region import RegionT2Ward, FeatStatEmpty
from ..space import get_ref, PointCloud


class SegGraph(nx.Graph):
    @property
    def ref(self):
        return next(iter(self.file_tree_dict.values())).ref

    @property
    def error(self):
        return sum(reg.t2_sq_error for reg in self.nodes)

    def __init__(self, file_tree_dict, ijk_set=None, _add_nodes=True,
                 **kwargs):
        """

        Args:
            obj (str): either 'ward' or 't2' (can also pass class, useful if
                       being called programatically)
            file_tree_dict (dict): keys are grp, values are FileTree
            ijk_set (set): restricts construction of nodes to this set of ijk
            _add_nodes (bool): toggles whether nodes are added, useful
                               internally if empty SegGraph needed
        """
        super().__init__()

        # get appropriate region constructor
        self.reg_type = RegionT2Ward

        # store file_tree_dict
        self.file_tree_dict = file_tree_dict

        # check that all file trees have same ref
        ref_list = [ft.ref for ft in file_tree_dict.values()]
        if any(ref_list[0] != ref for ref in ref_list[1:]):
            raise AttributeError('ref space mismatch')

        # ensure file_tree is loaded
        for ft in file_tree_dict.values():
            if not ft.ijk_fs_dict.keys():
                ft.load()

        if not _add_nodes:
            return

        # get ijk_set, intersection of all ijk in ft
        if ijk_set is None:
            ijk_set = (set(ft.ijk_fs_dict.keys()) for ft in
                       file_tree_dict.values())
            ijk_set = set.intersection(*ijk_set)

        # build regions
        for ijk in ijk_set:
            # space region occupies
            pc_ijk = PointCloud({tuple(ijk)}, ref=ref_list[0])

            # statistics of features in region
            fs_dict = dict()
            for grp, ft in file_tree_dict.items():
                fs_dict[grp] = ft.ijk_fs_dict[ijk]

            # build and store in graph
            reg = self.reg_type(pc_ijk=pc_ijk, fs_dict=fs_dict)
            self.add_node(reg)

        self.connect_neighbors(**kwargs)

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

    def from_file_tree_dict(self, file_tree_dict):
        # build map of old regions to new (those from new file_tree_dict)
        reg_map = {reg: reg.from_file_tree_dict(file_tree_dict)
                   for reg in self.nodes}

        # init new SegGraph
        sg = type(self)(obj=self.reg_type, file_tree_dict=file_tree_dict,
                        _add_nodes=False)

        # add edges which mirror original
        new_edges = ((reg_map[r0], reg_map[r1]) for r0, r1 in self.edges)
        sg.add_edges_from(new_edges)
        sg.add_nodes_from(reg_map.values())

        return sg

    def to_nii(self, f_out, ref, **kwargs):
        # load reference image
        ref = get_ref(ref)
        if ref.shape is None:
            raise AttributeError('ref must have shape')

        # build array
        x = self.to_array(shape=ref.shape, **kwargs)

        # save
        img_out = nib.Nifti1Image(x, ref.affine)
        img_out.to_filename(str(f_out))

        return img_out

    def to_array(self, fnc=None, fnc_include=None, shape=None, background=0):
        """ constructs array of mean feature per region """
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

        if shape is None:
            shape = reg_list[0].pc_ijk.ref.shape

        if set().intersection(*[r.pc_ijk for r in self]):
            raise AttributeError('non disjoint regions found')

        # build output array
        x = np.zeros(shape) * background
        for reg in reg_list:
            val = fnc(reg)
            for ijk in reg.pc_ijk:
                x[tuple(ijk)] += val

        return x

    def combine(self, reg_tuple):
        """ combines multiple regions into one

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

    def combine_by_reg(self, f_region, verbose=False):
        """ combines regions which share an idx in f_region (some parcellation)

        note: regions must all be single voxel at outset
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

        # combine
        reg_new_dict = dict()
        for reg_idx, reg_list in tqdm(reg_idx_reg_dict.items(),
                                      desc='combining',
                                      disable=not verbose):
            reg_new_dict[reg_idx] = self.combine(reg_list)

    def harmonize_via_add(self, apply=True):
        """ adds, uniformly, to each grp to ensure same average over whole reg

        note: the means meet at the weighted average of their means (more
        observations => smaller movement)

        Returns:
            mu_offset_dict (dict): keys are grp, values are offsets of average
        """

        # add together root nodes
        fs_dict = defaultdict(FeatStatEmpty)
        for reg in self.nodes:
            for grp, fs in reg.fs_dict.items():
                fs_dict[grp] += fs

        # sum different groups
        fs_all = sum(fs_dict.values())

        # build mu_offset_dict
        mu_offset_dict = {grp: fs_all.mu - fs.mu for grp, fs in
                          fs_dict.items()}

        # add to all regions
        if apply:
            for r in self.nodes:
                for grp, mu in mu_offset_dict.items():
                    r.fs_dict[grp].mu += mu
                r.reset()

        return mu_offset_dict

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
        sg = SegGraph(obj=self.reg_type, file_tree_dict=self.file_tree_dict,
                      _add_nodes=False)
        sg.file_tree_dict = self.file_tree_dict

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
