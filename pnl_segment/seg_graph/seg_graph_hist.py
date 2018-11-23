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

    def get_sig_hierarchical(self, alpha=.05):
        """ builds seg_graph of 'edge significant' reg

        todo: ref

        todo: no start @ root, start @ min pval ... can we holm these init reg?

        a region is 'edge significant' if it is significant but none of the
        parent regions which comprise it are.  if there are no comprising
        regions (ie it is a single voxel) then it is edge significant.

        Args:
            alpha (float): false positive rate

        Returns:
            sg_sig (SegGraph): all significant nodes
        """
        # get root nodes
        reg_root_list = [r for r in self.tree_history
                         if not nx.descendants(self.tree_history, r)]

        # adjust significance for root nodes
        p = alpha / len(reg_root_list)

        # build seg_graph
        sg_sig = SegGraph()
        sg_sig.obj_fnc_max = self.obj_fnc_max

        # build set of significant regions
        sig_reg_set = {(r, p) for r in reg_root_list if r.pval <= p}

        while sig_reg_set:
            # choose any significant region
            r, p = sig_reg_set.pop()

            # get list of its significant parents
            r_parents = list(self.tree_history.predecessors(r))
            p *= 1 / len(r_parents)
            r_parents = [r for r in r_parents if r.pval <= p]
            # todo: root nodes need not divide p

            if not r_parents:
                # if no significant parents, r is 'edge significant'
                sg_sig.add_node(r)
            else:
                # if any child is significant, add it to searchable set
                sig_reg_set |= {(r, p) for r in r_parents}

        return sg_sig

    def get_n(self, n):
        return list(self)[-n]

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

    def get_span_less_p_error(self, p=.9):
        """ gets the smallest (from hist) with at most p perc of current error

        Returns:
            seg_graph (SegGraph):
        """

        # create a list of all seg_graph in history
        sg_list = list(self)

        # err_max is error if considering entire volume as 1 region
        sg_smallest = sg_list[-1]
        err_thresh = sg_smallest.error * p

        # search from smallest to largest
        for sg in reversed(sg_list):
            if sg.error <= err_thresh:
                print(f'len of final sg: {len(sg)}')
                return sg

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

        for sg in self:
            size.append(len(sg))
            error.append(sg.error)

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
        # build seg_graph
        seg_graph = SegGraph()
        seg_graph.obj_fnc_max = self.obj_fnc_max

        # compute fnc + sort
        reg_list = []
        for r in self.tree_history.nodes:
            f = fnc(r)
            if f is None:
                # None acts as sentinel for invalid region
                continue
            reg_list.append((f, r))
        reg_list = sorted(reg_list, reverse=max)

        if not reg_list:
            return seg_graph

        # a list of all regions which contain some ijk value which has been
        # included in min_reg_list
        unspanned_reg = set(self.tree_history.nodes)

        # get list of spanning region with min fnc
        min_reg_list = list()
        while unspanned_reg and reg_list:
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

        # add nodes
        seg_graph.add_nodes_from(min_reg_list)

        return seg_graph
