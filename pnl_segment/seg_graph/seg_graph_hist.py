import networkx as nx
import nibabel as nib
import numpy as np

from .seg_graph import SegGraph


class SegGraphHistory(SegGraph):
    """ has a history of which regions were combined

    Attributes:
        tree_history (nx.Digraph): directed seg_graph, from component reg to sum
        reg_history_list (list): stores ordering of calls to combine()
    """

    def __init__(self):
        super().__init__()
        self.tree_history = nx.DiGraph()
        self.reg_history_list = list()

    def combine(self, reg_iter):
        reg_sum = super().combine(reg_iter)

        for reg in reg_iter:
            self.tree_history.add_edge(reg, reg_sum)

        self.reg_history_list.append(reg_sum)
        return reg_sum

    def get_max_fnc_array(self, fnc, ref):
        """ gets array which maximizes fnc

        every voxel belongs to many regions in tree_history, this assigns to
        each voxel the max value of fnc under all regions which contain it.

        Args:
            fnc (fnc): accepts regions, outputs scalar
            ref: reference space, has affine and shape

        Returns:
            x (array): array of volume
        """
        x = np.ones(ref.shape) * -np.inf

        for reg in self.tree_history.nodes:
            val = fnc(reg)
            for ijk in reg.pc_ijk:
                if x[ijk] < val:
                    x[ijk] = val

        return x

    def max_fnc_to_nii(self, fnc, f_out):
        """ builds array according get_max_fnc_array(), writes to nii
        """

        ref = next(iter(self.file_tree_dict.values())).ref
        x = self.get_max_fnc_array(fnc, ref)

        img = nib.Nifti1Image(x, ref.affine)
        img.to_filename(str(f_out))

    def __iter__(self):
        """ iterates through all version of seg_graph through its history

        NOTE: these part_graphs are not connected, neighboring regions do not
        have edges between them.
        """

        def build_part_graph(reg_set):
            # build seg_graph
            seg_graph = SegGraph()
            seg_graph.obj_fnc_max = np.inf
            seg_graph.add_nodes_from(reg_set)
            return seg_graph

        # get all leaf nodes (reg without ancestors in tree_history)
        reg_set = {r for r in self.tree_history.nodes
                   if not nx.ancestors(self.tree_history, r)}

        yield build_part_graph(reg_set)

        for reg_next in self.reg_history_list:
            # subtract the kids, add the next node
            reg_kids = set(self.tree_history.predecessors(reg_next))
            reg_set = (reg_set - reg_kids) | {reg_next}

            yield build_part_graph(reg_set)

    def get_min_error_span(self):
        """ returns the seg_graph which minimzes regularized error

        regularized error = error + (size - 1) / (max_size - 1) * max_error

        where size is the number of regions in the seg_graph, max_size is the
        most number of regions the seg_graph had (1 per voxel) and max_error
        is the maximum error (occurs at size = 1)

        Returns:
            seg_graph (SegGraph):
        """
        size = list()
        error = list()

        for seg_graph in self:
            size.append(len(seg_graph))
            error.append(sum(r.error for r in seg_graph.nodes))

        # find
        max_err = max(error)
        max_size = max(size)
        error_reg = [e + (s - 1) / (max_size - 1) * max_err
                     for e, s in zip(error, size)]

        min_reg_size = min(zip(error_reg, size))[1]

        for seg_graph in self:
            if len(seg_graph) == min_reg_size:
                return seg_graph

        raise RuntimeError('optimal seg_graph not found')

    def get_spanning_region(self, fnc, max=True):
        """ get subset of self.tree_history that covers with min fnc (greedy)

        by cover, we mean that each leaf in self.tree_history has exactly one
        region in the subset which is its descendant (dag points to root)

        Args:
            fnc (fnc): function to be minimized (accepts regions, gives obj
                       which can be ordered
            max (bool): toggles whether max fnc is chosen (otherwise min)
        """
        reg_list = self.tree_history.nodes
        f = [fnc(r) for r in reg_list]

        # sort a list of regions by fnc
        reg_list = sorted(zip(f, reg_list), reverse=max)

        # a list of all regions which contain some ijk value which has been
        # included in min_reg_list
        unspanned_reg = set(self.tree_history.nodes)

        # get list of spanning region with min fnc
        min_reg_list = list()
        while unspanned_reg:
            # get reg with min val
            _z, reg = reg_list.pop(0)

            if reg in unspanned_reg:
                # region has no intersection with any reg in min_reg_list
                # add to min_reg_list
                min_reg_list.append(reg)

                # rm its descendants and ancestors from unspanned_reg
                unspanned_reg -= {reg}
                unspanned_reg -= nx.descendants(self.tree_history, reg)
                unspanned_reg -= nx.ancestors(self.tree_history, reg)
            else:
                # region intersects some ijk which is already in min_reg_list
                continue

        # build seg_graph
        seg_graph = SegGraph()
        seg_graph.obj_fnc = self.obj_fnc
        seg_graph.obj_fnc_max = np.inf
        seg_graph.add_nodes_from(min_reg_list)

        return seg_graph