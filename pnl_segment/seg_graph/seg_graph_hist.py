import networkx as nx
import numpy as np

from .seg_graph import SegGraph


class SegGraphHistory(SegGraph):
    """ has a history of which regions were combined

    Attributes:
        tree_history (nx.Digraph): directed seg_graph, from component reg to sum
        reg_history_list (list): stores ordering of calls to combine()
    """

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
        reg_set = set(self.leaf_iter)

        yield build_part_graph(reg_set)

        for reg_next in self.reg_history_list:
            # subtract the kids, add the next node
            reg_kids = set(self.tree_history.predecessors(reg_next))
            reg_set = (reg_set - reg_kids) | {reg_next}

            yield build_part_graph(reg_set)

    @property
    def leaf_iter(self):
        return (r for r in self.tree_history.nodes
                if not nx.ancestors(self.tree_history, r))

    @property
    def root_iter(self):
        return (r for r in self.tree_history
                if not nx.descendants(self.tree_history, r))

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

    def cut_hierarchical(self, alpha=.05):
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

        # build seg_graph
        sg_sig = SegGraph()
        sg_sig.obj_fnc_max = self.obj_fnc_max

        # get number of voxels
        num_vox = sum(len(r) for r in self.root_iter)

        def is_sig(reg):
            return reg.pval <= (alpha * len(reg) / num_vox)

        def get_sig_predecessors(reg_iter):
            """
            Returns:
                reg_sig (set): set of all significnat predecessors
                reg_parent (set): set of all checked clusters
            """
            parent_iter = (set(self.tree_history.predecessors(r))
                           for r in reg_iter)
            reg_parent = set.union(*parent_iter)
            reg_sig = set(filter(is_sig, reg_parent))

            if reg_sig:
                _reg_sig, _reg_parent = get_sig_predecessors(reg_sig)
                reg_sig |= _reg_sig
                reg_parent |= _reg_parent
            return reg_sig, reg_parent

        root_sig = list(filter(is_sig, self.root_iter))
        if root_sig:
            reg_sig, reg_checked = get_sig_predecessors(root_sig)
            sg_sig.add_nodes_from(reg_sig)
        else:
            reg_checked = set()

        return sg_sig, reg_checked

    def cut_n(self, n):
        return list(self)[-n]

    def cut_max_fnc_array(self, fnc, ref):
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

    def cut_span_less_p_error(self, p=.9):
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

    def cut_min_error_span(self):
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

    def cut_spanning_region(self, fnc, max=True):
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

    def harmonize_via_add(self):
        """ adds, uniformly, to each grp to ensure same average over whole reg

        note: the means meet at the weighted average of their means (more
        observations => smaller movement)

        Returns:
            mu_offset_dict (dict): keys are grp, values are offsets of average
        """

        # add together root nodes
        fs_dict = sum(self.root_iter).fs_dict

        # sum different groups
        fs_all = sum(fs_dict.values())

        # build mu_offset_dict
        mu_offset_dict = {grp: fs_all.mu - fs.mu for grp, fs in
                          fs_dict.items()}

        # add to all regions
        all_reg = set(self.tree_history.nodes) | set(self.nodes)
        for r in all_reg:
            for grp, mu in mu_offset_dict.items():
                r.fs_dict[grp].mu += mu
            r.reset_obj()

        return mu_offset_dict
