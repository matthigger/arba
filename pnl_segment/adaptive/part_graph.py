import multiprocessing
import time
from collections import defaultdict

import mh_pytools.paralell
import networkx as nx
import nibabel as nib
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

from ..point_cloud import ref_space


class PartGraph(nx.Graph):
    def __init__(self, feat_label=None):
        # see PartGraphFactory to instantiate PartGraph
        super().__init__()
        self.obj_fnc = None
        self.obj_fnc_max = np.inf
        self._obj_edge_list = None
        self.feat_label = feat_label

    def to_img(self, f_out, ref, **kwargs):
        # load reference image
        ref = ref_space.get_ref(ref)
        if ref.shape is None:
            raise AttributeError('ref must have shape')

        # build array
        x = self.to_array(ref.shape, **kwargs)

        # save
        img_out = nib.Nifti1Image(x, ref.affine)
        img_out.to_filename(str(f_out))

        return img_out

    def to_array(self, shape, fnc=None, fnc_include=None, verbose=True):
        """ constructs array of mean feature per region """
        if fnc is None:
            if fnc_include is not None:
                nodes = [reg for reg in self.nodes if fnc_include(reg)]
            else:
                nodes = self.nodes
            reg_to_idx = {reg: idx for idx, reg in enumerate(nodes)}

            def fnc(reg):
                return reg_to_idx[reg]

        x = np.zeros(shape)
        tqdm_dict = {'desc': 'get mask per node', 'disable': not verbose}
        for reg in tqdm(self.nodes, **tqdm_dict):
            if fnc_include is None or fnc_include(reg):
                x += reg.pc_ijk.to_array(shape=shape) * fnc(reg)
        return x

    def combine(self, reg_iter):
        """ combines multiple regions into one

        Args:
            reg_iter (iter): iterator of regions to be combined
        """
        # create rew region
        reg_sum = sum(reg_iter)

        # get neighbor list (before removing any nodes)
        neighbr_set = frozenset().union(*[self.neighbors(r) for r in reg_iter])
        neighbr_set -= set(reg_iter)

        # rm old regions from graph
        for r in reg_iter:
            self.remove_node(r)

        # add new region to graph
        self.add_node(reg_sum)

        # add edges to all of combined nodes
        for r_neighbr in neighbr_set:
            self.add_edge(reg_sum, r_neighbr)

        return reg_sum

    def discard_zero_var(self):
        """ discards all regions which do not vary
        """
        for reg in list(self.nodes):
            if any(fs.var[0][0] == 0 for fs in reg.feat_stat):
                self.remove_node(reg)

    def combine_by_reg(self, f_region):
        """ combines regions which share an idx in f_region (some parcellation)

        note: regions must all be single voxel at outset
        """

        # find ijk_reg_dict, keys are ijk (tuple), values are reg they are in
        ijk_reg_dict = dict()
        for reg in self.nodes:
            if len(reg.pc_ijk) != 1:
                raise AttributeError('regions must consist of single vox')
            ijk = tuple(reg.pc_ijk.x[0, :])
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

        print(f'missing {len(ijk_missing)} regions (voxels)?')

        # combine
        reg_new_dict = dict()
        for reg_idx, reg_list in tqdm(reg_idx_reg_dict.items(),
                                      desc='combining'):
            reg_new_dict[reg_idx] = self.combine(reg_list)

    def reduce_to(self, num_reg_stop=1, edge_per_step=1, verbose=True,
                  par_thresh=10000, update_period=10):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (int): number of edges to remove per step (higher
                                 values are more computationally efficient and
                                 potentially less optimal)
            verbose (bool): toggles cmd line output
            par_thresh (int): min threshold for paralell computation of edge
                              weights
            update_period (float): how often command line updates are given

        Returns:
            obj_list (list): objective fnc at each combine
        """

        if len(self) < num_reg_stop:
            print(f'{len(self)} reg exist, cant reduce to {num_reg_stop}')

        if edge_per_step <= 0:
            raise AttributeError('edge_per_step must be positive')

        # init edges if need be
        if self._obj_edge_list is None:
            self._add_obj_edge_list(verbose=verbose)

        # init progress stats
        n_neigh_list = list()
        obj_list = []

        # init
        pbar = tqdm(total=len(self) - num_reg_stop,
                    desc='combining edges',
                    disable=not verbose)

        # combine edges until only n regions left
        len_init = len(self)
        last_update = time.time()
        while len(self) > num_reg_stop:
            t = time.time()

            # break early if no more valid edges available
            if not self._obj_edge_list or \
                    self._obj_edge_list[0][0] > self.obj_fnc_max:
                print(f'stop: no valid edges {len(self)} (obj:{num_reg_stop})')
                break

            # find n edges with min obj
            n = min(edge_per_step, len(self) - num_reg_stop)
            edge_list, _obj_list = self._get_min_n_edges(n)
            obj_list += _obj_list

            # combine them
            reg_list = list()
            for reg_set in edge_list:
                reg_list.append(self.combine(reg_set))

            # recompute obj of new edges to all neighbors of newly combined reg
            edge_list = list()
            for reg in reg_list:
                neighbor_list = list(self.neighbors(reg))
                n_neigh_list.append(len(neighbor_list))
                for reg_neighbor in neighbor_list:
                    edge_list.append((reg, reg_neighbor))
            self._add_obj_edge_list(edge_list, paralell=par_thresh)

            # command line update
            pbar.update((len_init - len(self)) - pbar.n)

            # output to command line(timing + debug)
            if verbose and time.time() - last_update > update_period:
                obj = np.mean(obj_list[-edge_per_step:])
                print(', '.join([f'n_edge: {len(self._obj_edge_list):1.2e}',
                                 f'time: {time.time() - t:.2f} sec',
                                 f'n_neighbors: {np.mean(n_neigh_list):1.2e}',
                                 f'obj: {obj:1.2e}']))
                last_update = time.time()
                n_neigh_list = list()

        return obj_list

    def _get_min_n_edges(self, n):
        # get edge_per_step edges with minimum objective
        edge_list = list()
        obj_list = list()
        while len(edge_list) < n:
            if not self._obj_edge_list or \
                    self._obj_edge_list[0][0] > self.obj_fnc_max:
                # no more edges
                break

            obj, (r1, r2) = self._obj_edge_list.pop(0)

            if r1 in self.nodes and r2 in self.nodes:
                obj_list.append(obj)
                edge_list.append(set((r1, r2)))

        if not edge_list:
            # no edges found at all
            return edge_list, obj_list

        # some edges may intersect each other, join these region sets
        edge_list_disjoint = []
        while edge_list:
            # get first reg_set
            reg_set = edge_list.pop(0)

            # find first intersection
            reg_set_int = next((r_set for r_set in edge_list if
                                r_set.intersection(reg_set)), None)

            if reg_set_int:
                # if it exists, add reg_set into the intersecting set
                reg_set_int |= reg_set
            else:
                # disjoint, add to disjoint list
                edge_list_disjoint.append(reg_set)

        return edge_list_disjoint, obj_list

    def _add_obj_edge_list(self, edge_list=None, paralell=False,
                           verbose=False):
        if edge_list is None:
            self._obj_edge_list = SortedList()
            edge_list = self.edges

        if not isinstance(paralell, bool):
            # a threshhold, not bool, was passed
            paralell = len(edge_list) > paralell

        if paralell:
            # compute (paralell) objective per edge
            pool = multiprocessing.Pool()
            res = pool.starmap_async(self.obj_fnc, edge_list)
            obj_list = mh_pytools.paralell.join(pool, res,
                                                desc='compute obj per edge (par)',
                                                verbose=verbose)

            # add to obj_edge_list
            for obj, reg_pair in zip(obj_list, edge_list):
                if obj < self.obj_fnc_max:
                    self._obj_edge_list.add((obj, reg_pair))

        else:
            # compute objective per edge
            tqdm_dict = {'desc': 'compute obj per edge',
                         'disable': not verbose}
            for reg_pair in tqdm(edge_list, **tqdm_dict):
                obj = self.obj_fnc(*reg_pair)
                if obj < self.obj_fnc_max:
                    self._obj_edge_list.add((obj, reg_pair))


class PartGraphHistory(PartGraph):
    """ has a history of which regions were combined """

    def __init__(self):
        super().__init__()
        self.tree_history = nx.DiGraph()

    def combine(self, reg_iter):
        reg_sum = super().combine(reg_iter)

        for reg in reg_iter:
            self.tree_history.add_edge(reg, reg_sum)

        return reg_sum

    def get_min_spanning_region(self, fnc):
        """ get subset of self.tree_history that covers with min fnc (greedy)

        by cover, we mean that each leaf in self.tree_history has exactly one
        region in the subset which is its descendant (dag points to root)

        Args:
            fnc (fnc): function to be minimized (accepts regions, gives obj
                       which can be ordered
        """

        # sort a list of regions by fnc
        reg_list = sorted([(fnc(reg), reg) for reg in self.tree_history.nodes])

        # a list of all regions which contain some ijk value which has been
        # included in min_reg_list
        unspanned_reg = set(self.tree_history.nodes)

        # get list of spanning region with min fnc
        min_reg_list = list()
        while unspanned_reg:
            # get reg with min val
            f, reg = reg_list.pop(0)

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

        # build part_graph
        pg = PartGraph()
        pg.obj_fnc = self.obj_fnc
        pg.obj_fnc_max = np.inf
        pg.add_nodes_from(min_reg_list)

        return pg
