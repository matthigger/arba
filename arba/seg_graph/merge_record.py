import pathlib
import tempfile

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.spatial.distance import dice

from .seg_graph import SegGraph
from ..region import RegionWardGrp
from ..space import PointCloud


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
    """

    def __init__(self, mask=None, pc=None, ref=None):
        super().__init__()

        assert (mask is None) != (pc is None), 'either mask xor pc required'
        if pc is None:
            pc = PointCloud.from_mask(mask)

        # define space
        self.ref = ref

        # record bijection from ijk to nodes (will be leafs in tree)
        self.ijk_leaf_dict = {ijk: idx for idx, ijk in enumerate(pc)}
        self.leaf_ijk_dict = {idx: ijk
                              for ijk, idx in self.ijk_leaf_dict.items()}

        self.add_nodes_from(self.leaf_ijk_dict.keys())

    def leaf_iter(self, node):
        """ returns an iterator over the leafs which make up node

        Args:
            node (int): node in self
        """
        assert node in self.nodes, 'node not found'

        node_set = {node}
        while node_set:
            _n = node_set.pop()
            node_list = list(self.neighbors(_n))
            if not node_list:
                # n is a leaf
                yield _n
            else:
                node_set |= set(node_list)

    def get_pc(self, node):
        """ identifies point_cloud associated with a node in tree_hist

        Args:
            node (int): node in self
        """
        ijk_iter = (self.leaf_ijk_dict[n] for n in self.leaf_iter(node))
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
            node_mask = self.get_pc(node).to_mask(shape=self.ref.shape)
            d = 1 - dice(mask.flatten(), node_mask.flatten())

            # store if max dice
            if d > d_max:
                node_min_dice = node
                d_max = d

        return node_min_dice, d_max

    def merge(self, reg_tuple=None, ijk_tuple=None):
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

        return node_sum

    def resolve_node(self, node, file_tree, split):
        """ gets the region associated with the node from the file_trees given

        Args:
            node (int): node
            file_tree (FileTree): file tree
            split (Split):

        Returns:
            reg (RegionWardT2): region
        """

        return RegionWardGrp.from_data(pc_ijk=self.get_pc(node),
                                       file_tree=file_tree,
                                       split=split)

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

    def to_nii(self, f_out, n=100, n_list=None):
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
            f_out = tempfile.NamedTemporaryFile(suffix='.nii.gz')
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
