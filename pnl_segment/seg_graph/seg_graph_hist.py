import networkx as nx
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

from .seg_graph import SegGraph


class SegGraphHistory(SegGraph):
    """ has a history of which regions were combined

    Attributes:
        tree_history (nx.Digraph): directed seg_graph, from component reg to
                                   sum.  "root" of tree is child, "leafs" are
                                   its farthest ancestors
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

    def from_file_tree_dict(self, file_tree_dict):
        sg_hist = super().from_file_tree_dict(file_tree_dict)

        # build map of old regions to new (those from new file_tree_dict)
        reg_map = {reg: reg.from_file_tree_dict(file_tree_dict)
                   for reg in self.tree_history.nodes}

        # add edges which mirror original
        new_edges = ((reg_map[r0], reg_map[r1])
                     for r0, r1 in self.tree_history.edges)
        sg_hist.tree_history.add_edges_from(new_edges)

        # map reg_history_list
        sg_hist.reg_history_list = [reg_map[r] for r in self.reg_history_list]

        return sg_hist

    def combine(self, reg_iter):
        reg_sum = super().combine(reg_iter)

        for reg in reg_iter:
            self.tree_history.add_edge(reg, reg_sum)

        self.reg_history_list.append(reg_sum)
        return reg_sum

    def get_maha_scaler(self, n=100):
        """ returns linear transform which fits scaled maha to chi2

        if samples were independent, scaled maha would be chi2

        Args:
            n (int): number of pts to sample in cdf

        Returns:
            lin_regress (LinearRegression): maps scaled maha to chi2
        """
        raise NotImplementedError

        # percentiles to match
        p = 1 - np.geomspace(.0001, 1, n)

        # cdf percentiles
        reg_any = next(iter(self.nodes))
        d = next(iter(reg_any.fs_dict.values())).d
        perc_chi2 = chi2.ppf(p, df=d)

        # observed percentiles
        scaled_maha = [r.maha * len(r) for r in self.tree_history]
        perc_obs = np.percentile(scaled_maha, p * 100)

        import matplotlib.pyplot as plt
        plt.plot(p, perc_chi2, label='chi2')
        plt.plot(p, perc_obs, label='obs')
        plt.legend()
        plt.show()

        # regress
        lr = LinearRegression()
        perc_chi2 = np.atleast_2d(perc_chi2)
        perc_obs = np.atleast_2d(perc_obs)
        lr.fit(perc_obs, perc_chi2)

        return lr

    def make_monotonic(self):
        """ ensures routes from composite regions to sums decrease in pval

        cut_hierarchical requires that the pval associated with a region is not
        just the probability that its specific volume shows significant
        difference between populations, but the probability that any sub-volume
        shows difference.  make_monotonic() changes pvalues to reflect the
        sub-volumes given by the region hierarchy
        """

        # iterate through regions from smallest to largest
        for reg in sorted(self.tree_history.nodes, key=len):

            # check each parent
            for reg_parent in self.tree_history.successors(reg):

                # if pvalue increases propagate child's pval to parent
                # (note: this pval can propagate to root given sort, fun!)
                if reg_parent.pval > reg.pval:
                    reg_parent.pval = reg.pval

    def cut_from_cut(self, sg_target):
        """ builds seg graph which corresponds (in space) per reg in sg_target
        """
        sg = SegGraph()
        sg.obj_fnc_max = self.obj_fnc_max

        for reg in sg_target.nodes:
            r = next(r for r in self.tree_history.nodes
                     if r.pc_ijk == reg.pc_ijk)
            sg.add_node(r)

        return sg

    def cut_hierarchical(self, spec=.05):
        """ builds seg_graph of 'edge significant' reg

        https://stat.ethz.ch/~nicolai/hierarchical.pdf
        Hierarchical Testing of Variable Importance
        Nicolai Meinshausen

        a region is 'edge significant' if it is significant but none of the
        parent regions which comprise it are.  if there are no comprising
        regions (ie it is a single voxel) then it is edge significant.

        Args:
            spec (float): false positive rate

        Returns:
            sg_sig (SegGraph): all significant nodes
        """

        # ensure monotonicity
        self.make_monotonic()

        # build seg_graph
        sg_sig = SegGraph()
        sg_sig.obj_fnc_max = self.obj_fnc_max

        # get number of voxels
        num_vox = sum(len(r) for r in self.root_iter)

        def is_sig(reg):
            c = num_vox / len(reg)
            return reg.pval * c <= spec

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

            # only regions which don't have any significant parents are minimal
            reg_sig_min = set()
            for reg in reg_sig:
                reg_par = set(self.tree_history.predecessors(reg))
                if not reg_par.intersection(reg_sig):
                    reg_sig_min.add(reg)

            sg_sig.add_nodes_from(reg_sig_min)
        else:
            reg_checked = set()
            reg_sig = set()

        return sg_sig, reg_sig, reg_checked

    def cut_n(self, n):
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

    def cut_less_p_error(self, p=.01):
        """ gets most coarse segmentation with error <= p * current error

        Returns:
            seg_graph (SegGraph):
        """
        if not 0 <= p <= 1:
            raise AttributeError('p must be in [0, 1]')

        err_thresh = self.error * p

        # search from smallest to largest
        sg_last = None
        for sg in self:
            if sg.error > err_thresh:
                if sg_last is None:
                    err_msg = f'no part_graph has < {p} of current error'
                    raise AttributeError(err_msg)
                break
            sg_last = sg
        return sg_last

    def cut_greedy_pval(self, alpha=.20):
        sg = SegGraph()
        sg.file_tree_dict = self.file_tree_dict

        reg_list = [r for r in self.tree_history.nodes if r.pval <= alpha]
        reg_list = sorted(reg_list, key=lambda r: r.pval)
        reg_covered = set()

        while reg_list:
            reg = reg_list.pop(0)
            if reg in reg_covered:
                continue
            else:
                sg.add_node(reg)

                reg_covered |= {reg}
                reg_covered |= nx.descendants(self.tree_history, reg)
                reg_covered |= nx.ancestors(self.tree_history, reg)

        return sg

    def cut_p_error_each_step(self, p=.3):
        """ 'splits' tree_history so long as error reduction is at least p

        Returns:
            seg_graph (SegGraph):
        """
        if not 0 < p < 1:
            raise AttributeError('p must be in [0, 1]')

        sg = self
        error = sg.error
        for sg_next in reversed(list(self)[:-1]):

            error_next = sg_next.error

            p_observed = (error - error_next) / error
            if p_observed < p:
                return sg

            sg = sg_next
            error = error_next

        raise RuntimeError

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

    def cut_spanning_region(self, fnc, max=True, n_region=None,
                            n_region_max=None):
        """ get subset of self.tree_history that covers with min fnc (greedy)

        by cover, we mean that each leaf in self.tree_history has exactly one
        region in the subset which is its descendant (dag points to root)

        Args:
            fnc (fnc): function to be minimized (accepts regions, gives obj
                       which can be ordered
            max (bool): toggles whether max fnc is chosen (otherwise min)
            n_region (int): maximum number of region (imposes that region sizes
                            are at least 1 / n_region of total size)
        """
        if n_region is None:
            min_size = 1
        else:
            size = sum(len(r) for r in self.root_iter)
            min_size = int(size / n_region)

        # build seg_graph
        seg_graph = SegGraph()
        seg_graph.obj_fnc_max = self.obj_fnc_max

        # compute fnc + sort
        reg_list = []
        for r in self.tree_history.nodes:
            if len(r) < min_size:
                continue

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
                if len(min_reg_list) == n_region_max:
                    break

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

    def harmonize_via_add(self, apply=True):
        """ adds, uniformly, to each grp to ensure same average over whole reg

        note: the means meet at the weighted average of their means (more
        observations => smaller movement)

        Returns:
            mu_offset_dict (dict): keys are grp, values are offsets of average
        """
        mu_offset_dict = super().harmonize_via_add(apply=False)

        # add to all regions
        if apply:
            node_set = set(self.tree_history.nodes) | set(self.nodes)
            for r in node_set:
                for grp, mu in mu_offset_dict.items():
                    r.fs_dict[grp].mu += mu
                r.reset()

        return mu_offset_dict
