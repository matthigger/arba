import networkx as nx

from .seg_graph import SegGraph
from ..region import RegionT2Ward
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

    def resolve_node(self, node, ft_dict):
        """ gets the region associated with the node from the file_trees given

        Args:
            node (int): node
            ft_dict (dict): keys are grp labels, values are FileTree

        Returns:
            reg (RegionWardT2): region
        """

        pc = self.get_pc(node)
        fs_dict = dict()
        for grp, ft in ft_dict.items():
            fs_dict[grp] = sum(ft.ijk_fs_dict[ijk] for ijk in pc)
        return RegionT2Ward(pc_ijk=pc, fs_dict=fs_dict)

    def get_iter_sg(self, ft_dict):
        """ iterator over seg_graph which undergoes recorded merge operations

        Args:
            ft_dict (dict): keys are grp labels, values are FileTree

        Yields:
            seg_graph (SegGraph): build from ft_hist
            reg_new (Region): the new region in seg_graph since last
            reg_last (tuple): the old regions which were merged since last
        """

        pc = PointCloud(self.leaf_ijk_dict.values(), ref=self.ref)
        ft0, ft1 = ft_dict.values()
        assert ft0.ref == ft1.ref, 'ft_dict doesnt have matching ref'
        assert ft0.ref == pc.ref, 'ft_dict doesnt match ref of merge_record'
        assert ft0.mask == ft1.mask, 'ft_dict doesnt have matching masks'
        assert PointCloud.from_mask(ft0.mask) == pc, 'ft_dict mask mismatch'

        # init seg graph
        sg = SegGraph(ft_dict=ft_dict, _add_nodes=False)

        # init node_reg_dict
        node_reg_dict = {node: self.resolve_node(node, ft_dict) for node in
                         self.leaf_ijk_dict.keys()}

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

    def get_node_reg_dict(self, ft_dict):
        """ maps each node to a region per ft_dict

        Args:
            ft_dict (dict): keys are grp labels, values are FileTree

        Returns:
            node_reg_dict (dict): keys are nodes, values are regions
        """

        iter_sg = self.get_iter_sg(ft_dict)

        sg, _, _ = next(iter_sg)

        node_reg_dict = dict()
        for reg in sg.nodes:
            ijk = next(iter(reg.pc_ijk))
            node = self.ijk_leaf_dict[ijk]
            node_reg_dict[node] = reg

        for _, reg, _ in iter_sg:
            node_reg_dict[len(node_reg_dict)] = reg

        return node_reg_dict

    def resolve_hist(self, ft_dict):
        """ returns a copy of tree_hist where each node is replaced by region

        NOTE: for large tree_hist, this will use a lot of memory

        Args:
            ft_dict (dict): keys are grp labels, values are FileTree

        Returns:
            tree_hist (nx.DiGraph): each node replaced with resolved version
            node_reg_dict (dict): keys are nodes, values are regions
        """
        node_reg_dict = self.get_node_reg_dict(ft_dict)
        tree_hist = nx.relabel_nodes(self,
                                     mapping=node_reg_dict,
                                     copy=True)

        return tree_hist, node_reg_dict
