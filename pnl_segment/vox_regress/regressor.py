import nibabel as nib
import numpy as np
from pnl_segment.point_cloud.ref_space import get_ref


class Regressor:
    """ learns and leverages linear relationships per voxel in nii imgs

    in other words we learn vector w which minimizes expected value of

    x @ w + w_0 = y

    where x is a feature vector (e.g. FA, MD) and y is vector of explanatory
    variable (e.g. age, sex, iq) etc

    Attributes:
        x_label (list): label of each dimension of x
        y_label (list): str label of each dimension of y, note that each key in
                       top level of sbj_img_tree must have each y_label
        sbj_img_tree (dict): 1st level: sbj objects, which have each of y_label
                                        as attributes.
                             2nd level: feat label, must be in x_feat
                             3rd level: (str or Path) to nii image
        sbj_mask_dict (dict): keys are sbj objects, values are str or Path to
                              masks of valid values

    For example:
        x_feat = ['fa', 'md']
        y_label = ['age', 'sex']
        sbj0 = {'age': 25, 'sex': 0}
        sbj1 = {'age': 30, 'sex': 1}
        sbj_img_tree = {sbj0: {'fa': 'sbj0_fa.nii.gz',
                              {'md': 'sbj0_md.nii.gz'},
                        sbj1: {'fa': 'sbj1_fa.nii.gz',
                               'md': 'sbj1_md.nii.gz'}}

    note: this object assumes input images are registered to common space as
    correspondence is established per ijk
    """

    @property
    def sbj_list(self):
        # always in same order on python3 (necessary)
        return list(self.sbj_img_tree.keys())

    def __init__(self, sbj_img_tree, y_label, x_label, sbj_mask_dict=dict()):
        self.x_label = x_label
        self.y_label = y_label
        self.sbj_img_tree = sbj_img_tree
        self.sbj_mask_dict = sbj_mask_dict

        # check ref space of each img
        sbj_rand = next(iter(sbj_img_tree.keys()))
        f_rand = next(iter(self.sbj_img_tree[sbj_rand].values()))
        self.ref_space = get_ref(f_rand)
        for sbj, d in self.sbj_img_tree.items():
            # check space on all x_feat img
            for f in d.values():
                if self.ref_space != get_ref(f):
                    raise AttributeError(f'ref space mismatch: {f}, {f_rand}')

            # check space on mask
            try:
                f = self.sbj_mask_dict[sbj]
                if self.ref_space != get_ref(f):
                    raise AttributeError(f'ref space mismatch: {f}, {f_rand}')
            except KeyError:
                # no mask given
                pass

    def learn(self):
        for ijk, x, y in self.ijk_x_y_iter():
            # learn

            # store
            pass

    def ijk_x_y_iter(self):
        """ iterates over each voxel

        Returns:
            ijk (tuple): ijk position
            x (np.array): x features
            y (np.array): y features
        """
        # get Y for each sbj
        n_sbj = len(self.sbj_list)
        y = np.ones((n_sbj, len(self.y_label))) * np.nan
        for sbj_idx, sbj in enumerate(self.sbj_list):
            y[sbj_idx, :] = self.get_y_array(sbj)

        # get mask datacube (space0, space1, space2, n_sbj)
        mask = np.stack((self._get_mask_array(sbj) for sbj in self.sbj_list),
                        axis=3)

        # get datacube x (space0, space1, space2, len(self.x_label), n_sbj)
        x_all = np.stack([self.get_x_array(sbj) for sbj in self.sbj_list],
                         axis=4)

        for ijk, _ in np.ndenumerate(mask[:, :, :, 0]):
            sbj_mask_vec = mask[ijk[0], ijk[1], ijk[2], :].astype(bool)
            if not any(sbj_mask_vec):
                continue
            x = x_all[ijk[0], ijk[1], ijk[2], :, sbj_mask_vec]
            yield ijk, x, y[sbj_mask_vec, :]
        raise StopIteration

    def get_x_array(self, sbj):
        """ loads data from nii files, returns array

        Args:
            sbj: key to self.sbj_img_tree

        Returns:
            x (np.array): (n_i, n_j, n_k, len(self.x_label)) features from img
        """
        x_list = list()
        for feat in self.x_label:
            img = nib.load(str(self.sbj_img_tree[sbj][feat]))
            x_list.append(img.get_data())
        return np.stack(x_list, axis=3)

    def get_y_array(self, sbj):
        y = np.ones(len(self.y_label)) * np.nan
        for y_idx, y_str in enumerate(self.y_label):
            y[y_idx] = getattr(sbj, y_str)
        return y

    def _get_mask_array(self, sbj):
        """ gets mask (builds array of ones if sbj not in self.sbj_mask_dict)
        Args:
            sbj
        Returns:
            mask (np.array): mask
        """
        try:
            # load and return mask
            f = self.sbj_mask_dict[sbj]
            img = nib.load(str(f))
            return img.get_data()
        except KeyError:
            # no mask given for sbj
            return np.ones(shape=self.ref_space.shape)

    def map_to(self):
        """ maps images to a set of y_feat """
        raise NotImplementedError
