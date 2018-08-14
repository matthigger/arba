from collections import defaultdict

import nibabel as nib
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering

from pnl_segment.point_cloud_ijk import PointCloudIJK


class Clusterer:
    """ a dict of point clouds, has methods for clustering and outputting
    """

    def __init__(self, f_dict, target_idx):
        self.ijk_dict = {sbj: PointCloudIJK.from_mask_nii(f, target_idx)
                         for sbj, f in f_dict.items()}
        self.xyz_dict = {sbj: ijk.to_xyz()
                         for sbj, ijk in self.ijk_dict.items()}

    def __call__(self, k, center=True, method='kmeans_mini'):
        """ k means clustering on xyz of all points

        Args:
            k (int): how many clusters to create
            center (bool): toggles centering of xyz per label pre-clustering
            method (str): 'kmeans_mini', 'kmeans', 'ward'

        Returns:
            pc_dict (Clusterer): keys are (label, idx), values are pc.
                                      note that all pts in self[label] are
                                      represented in union of (label, idx) for
                                      idx = 0, ..., k - 1
        """
        # sbj_list tracks which pts are from which label
        # [('sbj1', 10), ('sbj2', 20)] means first 10 from sbj1, ...
        sbj_list = [(label, len(pc)) for label, pc in self.ijk_dict.items()]

        # aggregate all points in xyz
        pc_xyz = [self.xyz_dict[sbj] for sbj, _ in sbj_list]
        xyz = np.vstack([pc.x - pc.center * center for pc in pc_xyz])

        # init clustering obj
        cluster_dict = {'kmeans_mini': (MiniBatchKMeans, {}),
                        'kmeans': (KMeans, {}),
                        'ward': (AgglomerativeClustering, {'linkage': 'ward'})}
        cluster_class, class_kwargs = cluster_dict[method]
        cluster_obj = cluster_class(n_clusters=int(k), **class_kwargs)

        # cluster
        cluster_idx = cluster_obj.fit_predict(xyz)

        # build dict of point clouds by cluster
        pc_clust_dict = PointCloudClusterDict(k)
        last_idx = 0
        for sbj, n in sbj_list:
            pc_ijk = self.ijk_dict[sbj]

            # get cluster idx labels associated with pc_ijk
            c_idx = cluster_idx[last_idx: last_idx + n]
            last_idx += n

            # build point clouds which partition pc_ijk
            for idx in range(int(k)):
                ijk = pc_ijk.x[c_idx == idx, :]

                pc = PointCloudIJK(ijk, affine=pc_ijk.affine)

                # we use 1, 2, ..., k to label subregions as 0 is background
                pc_clust_dict[sbj][idx + 1] = pc

        return pc_clust_dict


class PointCloudClusterDict(defaultdict):
    """ output of PointCloudDict.cluster(), with convenience methods """

    def __init__(self, k):
        super().__init__(PointCloudCluster)
        self.k = k


class PointCloudCluster(dict):
    def to_array(self, f_ref, dtype=None, idx_error=True):
        """ builds mask array from output of self.cluster()

        Args:
            f_ref (str or Path): reference file
            dtype (datatype): array type
            idx_error (bool): toggles raise flag on out of bounds

        Returns:
            x (array): array
        """

        # choose dtype as most compact representation which handles all k
        if dtype is None:
            k = max(self.keys())
            if k <= 255:
                dtype = np.uint8
            else:
                dtype = np.uint16

        # get shape
        shape = nib.load(str(f_ref)).shape

        # export to array per pc
        x = np.zeros(shape, dtype=dtype)
        for idx, pc in self.items():
            for ijk_row in pc.x:
                try:
                    x[tuple(ijk_row)] = idx
                except IndexError:
                    # ijk_row is outside of x.shape
                    if idx_error:
                        raise IndexError
            x = np.ma.masked_where(x < 1, x)

        # check for overlap
        n_pts_out = len(x.compressed())
        assert n_pts_out == sum([len(pc) for pc in self.values()]), \
            'overlap detected'

        return x

    def to_file(self, f_out, **kwargs):
        """ writes file from output of self.cluster()

        Args:
            f_out (str or pathlib.Path): file out
        """
        # get output array
        x = self.to_array(**kwargs)

        # write it to file
        affine = next(iter(self.values())).affine
        img_pc_dict = nib.Nifti1Image(x, affine)
        img_pc_dict.to_filename(str(f_out))
