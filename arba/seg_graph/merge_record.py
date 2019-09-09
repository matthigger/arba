import pathlib
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import seaborn as sns
from scipy.spatial.distance import dice
from sortedcontainers.sortedset import SortedSet

from .seg_graph import SegGraph
from ..space import PointCloud, Mask


class MergeRecord(nx.DiGraph):
    """ records all merges which have happened to a seg_graph

    graph is directed from large nodes to smaller ones

    this object manages only the spaces which have been merged, no statistics
    (or specific files / sbjs etc) are maintained.  It has methods of
    reproducing the merge history from data objects

    Attributes:
        ref: reference space
        ijk_leaf_dict (dict): keys are ijk, values are leaf nodes
        leaf_ijk_dict (dict): keys are leaf nodes, values are ijk
        fnc_dict (dict): keys are function names (str), values are fnc applied
                         to the region objects
        fnc_node_val_list (dict): keys are function names (fnc_dict.keys()),
                                  values are dicts (whose keys are nodes and
                                  values are fnc returns)
        node_size_dict (dict): keys are nodes, values are size (in voxels)
    """

    def __init__(self, mask=None, pc=None, ref=None, fnc_dict=None, **kwargs):
        super().__init__()

        if pc is None:
            if mask is None:
                pc = set()
            else:
                pc = PointCloud.from_mask(mask)

        self.fnc_dict = fnc_dict
        if fnc_dict is None:
            self.fnc_node_val_list = dict()
        else:
            self.fnc_node_val_list = {var_name: dict()
                                      for var_name in fnc_dict.keys()}

        # define space
        self.ref = ref

        # record bijection from ijk to nodes (will be leafs in tree)
        self.ijk_leaf_dict = {ijk: idx for idx, ijk in enumerate(pc)}
        self.leaf_ijk_dict = {idx: ijk
                              for ijk, idx in self.ijk_leaf_dict.items()}

        self.add_nodes_from(self.leaf_ijk_dict.keys())

        self.node_size_dict = {n: 1 for n in self.leaf_ijk_dict.keys()}

    def _cut_biggest_rep(self, node_val_dict, thresh=.9):
        """ gets largest nodes whose val is >= thresh% of leaf ancestors mean

        Args:
            node_val_dict (dict): keys are nodes, values are floats
            thresh (float): threshold of leaf ancestor mean needed for a node
                            to be a valid representative

        Returns:
            node_list (list):
        """

        valid_node_set = set()
        node_sum_dict = defaultdict(lambda: 0)
        node_count_dict = defaultdict(lambda: 0)
        for n in range(len(self)):
            # the iterator above yields lexicographical sorted nodes

            kids = list(self.neighbors(n))

            if kids:
                # non-leaf node

                # update counts of node_sum and node_count
                node_sum_dict[n] = sum(node_sum_dict[_n] for _n in kids)
                node_count_dict[n] = sum(node_count_dict[_n] for _n in kids)

                # if node's value exceeds threshold, add it to valid nodes
                _thresh = thresh * node_sum_dict[n] / node_count_dict[n]
                if node_val_dict[n] >= _thresh:
                    valid_node_set.add(n)
            else:
                # leaf node, valid by default (has no constituents)
                valid_node_set.add(n)
                node_sum_dict[n] = node_val_dict[n]
                node_count_dict[n] = 1

        # get biggest valid nodes
        node_list = list()
        for n in reversed(range(len(self))):
            # reverse lexicographical
            if n in valid_node_set:
                node_list.append(n)
                valid_node_set -= set(nx.descendants(self, n))

        return node_list

    def get_cover(self, node_list, sort_flag=True):

        if sort_flag:
            # sorts from biggest to smallest
            node_list = sorted(node_list, reverse=True)

        # init
        node_covered = set()
        cover = list()

        while node_list:
            n = node_list.pop(0)
            if n in node_covered:
                continue
            else:
                # add reg to significant regions
                cover.append(n)

                # add all intersecting regions to reg_covered (no need to add
                # n, its only in node_list_sorted once)
                node_covered |= nx.descendants(self, n)
                node_covered |= nx.ancestors(self, n)

        return cover

    def _cut_greedy(self, node_val_dict, max_flag=True):
        """ gets node of disjoint reg which minimize val

        NOTE: the resultant node_list covers node_val_dict, ie each node in
        node_val_dict has some ancestor, descendant or itself in node_list

        Args:
            node_val_dict (dict): keys are nodes, values are associated values
                                  to be minimized
            max_flag (bool): toggles max or min

        Returns:
             node_list (list): nodes have minimum val, are disjoint
        """
        node_list = sorted(node_val_dict.keys(), key=node_val_dict.get,
                           reverse=max_flag)

        return self.get_cover(node_list=node_list, sort_flag=False)

    def apply_fnc_leaf(self, sg):
        for reg in sg.nodes:
            ijk = next(iter(reg.pc_ijk))
            node = self.ijk_leaf_dict[ijk]
            for fnc_name, fnc in self.fnc_dict.items():
                val = fnc(reg)
                self.fnc_node_val_list[fnc_name][node] = val

    def leaf_iter(self, node=None, node_list=None):
        """ returns an iterator over the leafs which cover a node (or nodes)

        Args:
            node (int): node in self
            node_list (list): nodes in self

        Yields:
            leaf (int): node in self which is a descendant leaf of node (or
                        some node in node_list)
        """
        assert (node is None) != (node_list is None), 'node xor node_list'
        assert node in self.nodes, 'node not found'

        if node_list is None:
            node_list = [node]

        node_list = SortedSet(node_list)
        while node_list:
            # get largest node
            node = node_list.pop()
            neighbor_list = list(self.neighbors(node))
            if not neighbor_list:
                # n is a leaf
                yield node
            else:
                node_list |= set(neighbor_list)

    def get_pc(self, **kwargs):
        """ identifies point_cloud associated with a node in tree_hist
        """

        ijk_iter = (self.leaf_ijk_dict[n] for n in self.leaf_iter(**kwargs))
        pc = PointCloud(ijk_iter, ref=self.ref)
        assert len(pc), 'empty space associated with node?'
        return pc

    def get_largest_node(self, ijk=None, n=None):
        """ gets largest node which contains ijk or n

        Args:
            ijk (tuple): voxel
            n (int): node

        Returns:
            n_out (int): node which contains ijk or n
        """
        assert (ijk is None) != (n is None), 'either ijk xor n'

        if n is None:
            n = self.ijk_leaf_dict[ijk]

        # climb tree to largest node
        while True:
            _n = next(self.predecessors(n), None)

            if _n is None:
                return n
            else:
                n = _n

    def get_node_max_dice(self, mask):
        pc = PointCloud.from_mask(mask)
        node_set = set()
        for ijk in pc:
            # get node
            node = self.ijk_leaf_dict[ijk]

            # add node and all
            node_set.add(node)
            node_set |= nx.ancestors(self, node)

        assert pc, 'no intersection with mask'

        d_max = 0
        for node in node_set:
            # compute dice of the node
            node_mask = self.get_pc(node=node).to_mask(shape=self.ref.shape)
            d = 1 - dice(mask.flatten(), node_mask.flatten())

            # store if max dice
            if d > d_max:
                node_min_dice = node
                d_max = d

        return node_min_dice, d_max

    def merge(self, reg_tuple=None, ijk_tuple=None, reg_sum=None):
        """ records merge() operation

        Args:
            reg_tuple (tuple): regions to be merged, each has a pc_ijk
                               attribute
            ijk_tuple (tuple): representative voxels to be merged, note that
                               only one voxel per region is needed

        Returns:
            node_sum (int): new node
        """
        assert (reg_tuple is None) != (ijk_tuple is None), \
            'reg_tuple xor ijk_tuple'

        # get ijk_tuple
        if ijk_tuple is None:
            ijk_tuple = tuple(next(iter(reg.pc_ijk)) for reg in reg_tuple)

        # look up nodes
        node_tuple = tuple(self.get_largest_node(ijk=ijk) for ijk in ijk_tuple)

        # get new node idx
        node_sum = len(self.nodes)

        # add new edges in tree_hist
        for node in node_tuple:
            self.add_edge(node_sum, node)

        # compute fnc on sum
        for fnc_name, fnc in self.fnc_dict.items():
            assert reg_sum is not None, 'reg_sum required if fnc'
            val = fnc(reg_sum, reg_tuple=reg_tuple)
            self.fnc_node_val_list[fnc_name][node_sum] = val

        # track size
        self.node_size_dict[node_sum] = \
            sum(self.node_size_dict[n] for n in node_tuple)

        return node_sum

    def resolve_node(self, node, file_tree, reg_cls):
        """ gets the region associated with the node from the file_trees given

        Args:
            node (int): node
            file_tree (FileTree): file tree

        Returns:
            reg (RegionWardT2): region
        """

        return reg_cls.from_file_tree(pc_ijk=self.get_pc(node=node),
                                      file_tree=file_tree)

    def resolve_pc(self, pc):
        """ gets node associated with point cloud
        """
        raise NotImplementedError('untested')

        _pc = pc

        ijk = next(iter(pc))
        node = self.ijk_leaf_dict[ijk]
        ijk_cover = {ijk}

        while _pc - ijk_cover:
            node = next(self.predecessors(node))
            ijk_cover = set(self.leaf_iter(node))
            assert not (ijk_cover - pc), 'invalid point cloud'

        return node

    def get_iter_sg(self, file_tree, split):
        """ iterator over seg_graph which undergoes recorded merge operations

        Args:
            file_tree (FileTree): file tree
            split (Split):

        Yields:
            seg_graph (SegGraph): build from ft_hist
            reg_new (Region): the new region in seg_graph since last
            reg_last (tuple): the old regions which were merged since last
        """

        # init seg graph
        sg = SegGraph(file_tree=file_tree, split=split, _add_nodes=False)

        # init node_reg_dict
        node_reg_dict = {node: self.resolve_node(node, file_tree, split)
                         for node in self.leaf_ijk_dict.keys()}

        # add leafs to sg
        sg.add_nodes_from(node_reg_dict.values())
        sg.connect_neighbors()

        yield sg, None, None

        # get latest node, merge it
        node_new = len(self.leaf_ijk_dict)

        while node_new in self.nodes:
            # find last regions, build new region sum
            reg_last = tuple(node_reg_dict[n]
                             for n in self.neighbors(node_new))
            reg_new = sg.merge(reg_last)

            # update node_reg_dict
            node_reg_dict[node_new] = reg_new

            # delete old nodes from dict to save memory
            for n in self.neighbors(node_new):
                del node_reg_dict[n]

            yield sg, reg_new, reg_last

            node_new += 1

    def get_node_reg_dict(self, file_tree, split):
        """ maps each node to a region per file_tree, split

        Args:
            file_tree (FileTree): file tree
            split (Split):

        Returns:
            node_reg_dict (dict): keys are nodes, values are regions
        """

        iter_sg = self.get_iter_sg(file_tree, split)

        sg, _, _ = next(iter_sg)

        node_reg_dict = dict()
        for reg in sg.nodes:
            ijk = next(iter(reg.pc_ijk))
            node = self.ijk_leaf_dict[ijk]
            node_reg_dict[node] = reg

        for _, reg, _ in iter_sg:
            node_reg_dict[len(node_reg_dict)] = reg

        return node_reg_dict

    def resolve_hist(self, file_tree, split):
        """ returns a copy of tree_hist where each node is replaced by region

        NOTE: for large tree_hist, this will use a lot of memory

        Args:
            file_tree (FileTree): file tree
            split (Split):

        Returns:
            tree_hist (nx.DiGraph): each node replaced with resolved version
            node_reg_dict (dict): keys are nodes, values are regions
        """
        node_reg_dict = self.get_node_reg_dict(file_tree, split)
        tree_hist = nx.relabel_nodes(self,
                                     mapping=node_reg_dict,
                                     copy=True)

        return tree_hist, node_reg_dict

    def iter_node_pc_dict(self, as_mask=False):
        node_pc_dict = {leaf: PointCloud({ijk})
                        for leaf, ijk in self.leaf_ijk_dict.items()}

        if as_mask:
            node_pc_dict = {n: pc.to_mask(shape=self.ref.shape)
                            for n, pc in node_pc_dict.items()}

        yield node_pc_dict

        node_next = len(node_pc_dict)
        while node_next in self.nodes:
            children = self.neighbors(node_next)

            # build pc_next
            pc_next = PointCloud({}, ref=self.ref)
            if as_mask:
                pc_next = pc_next.to_mask(shape=self.ref.shape)

            for n in children:
                if as_mask:
                    pc_next += node_pc_dict[n]
                else:
                    pc_next |= node_pc_dict[n]
                del node_pc_dict[n]

            node_pc_dict[node_next] = pc_next.astype(bool)

            yield node_pc_dict
            node_next += 1

    def to_nii(self, f_out=None, n=10, n_list=None):
        """ writes a 4d volume with at different granularities

        Args:
            f_out (str or Path): output file
            n (int): number of granularities to output.  defaults to 100
            n_list (list): explicitly pass the granularities to output

        Returns:
             f_out (Path): a 4d nii volume, first 3d are space.  the 4th
             n_list (array): number
        """
        assert (n is None) != (n_list is None), 'either n xor n_list required'

        # get n_list
        if n_list is None:
            n_init = len(self.ijk_leaf_dict)
            n_last = len(
                [n for n in self.nodes if not any(self.predecessors(n))])
            n_list = np.geomspace(n_init, n_last, min(n, n_init - n_last))
        n_list = sorted(set([int(n) for n in n_list]), reverse=True)

        # get f_out
        if f_out is None:
            f_out = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
            f_out = pathlib.Path(f_out)

        # build array
        shape = (*self.ref.shape, len(n_list))
        x = np.zeros(shape)
        for n_idx, n in enumerate(n_list):
            for node_mask_dict in self.iter_node_pc_dict(as_mask=True):
                if len(node_mask_dict) == n:
                    x[:, :, :, n_idx] = sum(n * mask for
                                            n, mask in node_mask_dict.items())

        # write to nii
        img = nib.Nifti1Image(x, self.ref.affine)
        img.to_filename(str(f_out))

        return f_out, n_list

    def plot_size_v(self, fnc, label=None, mask=None, log_y=False):
        if label is None:
            label = fnc.__name__

        num_nodes = len(self.nodes)

        sns.set(font_scale=1.2)
        if mask is None:
            node_color = None
        else:
            # compute node_color
            node_color = dict()
            for node in range(num_nodes):
                if node < len(self.leaf_ijk_dict.keys()):
                    ijk = self.leaf_ijk_dict[node]
                    node_color[node] = mask[ijk].astype(bool)
                else:
                    n0, n1 = list(self.neighbors(node))
                    s0, s1 = self.node_size_dict[n0], self.node_size_dict[n1]
                    c0, c1 = node_color[n0], node_color[n1]
                    lam0 = s0 / (s0 + s1)
                    node_color[node] = c0 * lam0 + c1 * (1 - lam0)

        node_pos = dict()
        for n, size in self.node_size_dict.items():
            if isinstance(fnc, dict):
                node_pos[n] = size, fnc[n]
            else:
                node_pos[n] = size, self.fnc_node_val_list[fnc][n]

        nodelist = [n for n, c in node_color.items() if c > 0]
        node_color = {n: node_color[n] for n in nodelist}
        nx.draw_networkx_nodes(self, nodelist=nodelist, pos=node_pos,
                               node_color=np.array(list(node_color.values())),
                               vmin=0, vmax=1, cmap=plt.get_cmap('bwr'))
        nx.draw_networkx_edges(self, pos=node_pos)
        plt.xlabel('size')
        plt.ylabel(label)

        ax = plt.gca()
        if log_y:
            ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(left=1, right=num_nodes)

    def build_mask(self, node_list):
        # init empty mask
        assert self.ref.shape is not None, 'ref must have shape'
        mask = Mask(np.zeros(self.ref.shape), ref=self.ref).astype(int)

        # sort nodes from biggest (value + space) to smallest
        node_set = SortedSet(node_list)
        while node_set:
            node = node_set.pop()

            # remove all the nodes which would be covered by node
            node_set -= set(nx.descendants(self, node))

            # record position of node
            for ijk in self.get_pc(node=node):
                mask[ijk] = node

        return mask