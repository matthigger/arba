from collections import defaultdict

import networkx as nx
import nibabel as nib
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..region import FeatStatEmpty
from ..space import get_ref


class SegGraph(nx.Graph):
    @property
    def ref(self):
        return next(iter(self.file_tree_dict.values())).ref

    @property
    def error(self):
        return sum(reg.sq_error for reg in self.nodes)

    def __init__(self):
        # see factory, use of __init__ directly is discouraged
        super().__init__()
        self.file_tree_dict = None

    def from_file_tree_dict(self, file_tree_dict):
        # build map of old regions to new (those from new file_tree_dict)
        reg_map = {reg: reg.from_file_tree_dict(file_tree_dict)
                   for reg in self.nodes}

        # init new sg_hist
        sg = type(self)()
        sg.file_tree_dict = file_tree_dict

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
        reg_sum = sum(reg_tuple)

        # get neighbor list (before removing any nodes)
        neighbr_set = frozenset().union(
            *[self.neighbors(r) for r in reg_tuple])
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

    def get_sig(self, alpha=.05, method='holm'):
        """ returns a SegGraph containing only significant regions in self

        Args:
            alpha (float): significance level
            method (str): see multipletests, defaults to Holm's

        Returns:
            sg (SegGraph): contains only significant regions
        """

        reg_list = list(self.nodes)
        pval_list = [reg.pval for reg in reg_list]
        if pval_list:
            is_sig_vec = multipletests(pval_list, alpha=alpha, method=method)[
                0]
            reg_sig_list = [r for r, is_sig in zip(reg_list, is_sig_vec) if
                            is_sig]
        else:
            reg_sig_list = list()

        # init seg graph
        sg = SegGraph()
        sg.file_tree_dict = self.file_tree_dict

        # add sig regions
        sg.add_nodes_from(reg_sig_list)

        return sg
