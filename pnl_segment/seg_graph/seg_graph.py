import multiprocessing
import time
from collections import defaultdict

import networkx as nx
import nibabel as nib
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

import mh_pytools.parallel
from ..region import Region
from ..space import get_ref


class SegGraph(nx.Graph):
    @property
    def error(self):
        return sum(reg.error for reg in self)

    def __init__(self):
        # see factory, use of __init__ directly is discouraged
        super().__init__()
        self._obj_edge_list = None
        self.file_tree_dict = None
        self.obj_fnc_max = np.inf

    def to_nii(self, f_out, ref, **kwargs):
        # load reference image
        ref = get_ref(ref)
        if ref.shape is None:
            raise AttributeError('ref must have shape')

        # build array
        x = self.to_array(**kwargs)

        # save
        img_out = nib.Nifti1Image(x, ref.affine)
        img_out.to_filename(str(f_out))

        return img_out

    def to_array(self, fnc=None, fnc_include=None, overlap=False, shape=None):
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

        # build output array
        x = np.zeros(shape)
        for reg in reg_list:
            val = fnc(reg)
            for _x in reg.pc_ijk:
                if not overlap and x[tuple(_x)] != 0:
                    raise AttributeError('regions are not disjoint')
                x[tuple(_x)] += val

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

        # rm old regions from seg_graph
        for r in reg_iter:
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

    def reduce_to(self, num_reg_stop=1, edge_per_step=None, verbose=True,
                  par_thresh=False, update_period=10):
        """ combines neighbor nodes until only num_reg_stop remain

        Args:
            num_reg_stop (int): number of unique regions @ stop
            edge_per_step (float): (0, 1) how many edges (of those remaining)
                                   to combine in each step.  if not passed 1
                                   edge is combined at all steps.
            verbose (bool): toggles cmd line output
            par_thresh (int): min threshold for paralell computation of edge
                              weights
            update_period (float): how often command line updates are given

        Returns:
            obj_list (list): objective fnc at each combine
        """

        if len(self) < num_reg_stop:
            print(f'{len(self)} reg exist, cant reduce to {num_reg_stop}')

        if edge_per_step is not None and not (0 < edge_per_step < 1):
            raise AttributeError(
                'edge_per_step not in (0, 1): {edge_per_step}')

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
        n = 1
        while len(self) > num_reg_stop:
            # break early if no more valid edges available
            if not self._obj_edge_list or \
                    self._obj_edge_list[0][0] > self.obj_fnc_max:
                print(f'stop: no valid edges {len(self)} (obj:{num_reg_stop})')
                break

            # find n edges with min obj
            if edge_per_step is not None:
                n = np.ceil(len(self) * edge_per_step).astype(int)
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
                obj = np.mean(obj_list[-n:])
                print(', '.join([f'n_edge: {len(self._obj_edge_list):1.2e}',
                                 f'time: {time.time() - last_update:.2f} sec',
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
            raise NotImplementedError
            pool = multiprocessing.Pool()
            res = pool.starmap_async(self.obj_fnc, edge_list)
            obj_list = mh_pytools.parallel.join(pool, res,
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
                obj = Region.get_error_delta(*reg_pair)
                if obj < self.obj_fnc_max:
                    self._obj_edge_list.add((obj, reg_pair))

    def harmonize_via_add(self, apply=True):
        """ adds, uniformly, to each grp to ensure same average over whole reg

        note: the means meet at the weighted average of their means (more
        observations => smaller movement)

        Returns:
            mu_offset_dict (dict): keys are grp, values are offsets of average
        """

        # add together root nodes
        fs_dict = sum(self.nodes).fs_dict

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
