import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from pnl_segment.point_cloud.ref_space import get_ref


class PolyRegressor:
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

    def __init__(self, sbj_img_tree, y_label, x_label, sbj_mask_dict=dict(),
                 degree=2):
        self.x_label = x_label
        self.y_label = y_label
        self.sbj_img_tree = sbj_img_tree
        self.sbj_mask_dict = sbj_mask_dict
        self.poly = PolynomialFeatures(degree=degree)
        self.ijk_regress_dict = dict()

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

        # init r2 score
        self.r2_score = np.ones(self.ref_space.shape) * np.nan
        self.obs_to_var = np.ones(self.ref_space.shape) * np.nan

    def fit(self, verbose=False, obs_to_var_thresh=10):
        """ fits polynomial per voxel

        Args:
            verbose (bool): toggles command line output
            obs_to_var_thresh (int): ratio of observation to variables required
                                    (minimum) for regression to be performed

        https://stats.stackexchange.com/questions/29612/minimum-number-of-observations-for-multiple-linear-regression
        """
        tqdm_dict = {'disable': not verbose,
                     'desc': 'fit per vox'}
        for ijk, x, y in tqdm(self.ijk_x_y_iter(), **tqdm_dict):
            # learn
            x_poly = self.poly.fit_transform(x)

            obs_to_var = x_poly.shape[0] / x_poly.shape[1]
            if obs_to_var < obs_to_var_thresh:
                # more samples than # of parameters required
                continue

            r = LinearRegression(copy_X=False, n_jobs=-1, fit_intercept=False)
            r.fit(x_poly, y)

            # store
            self.ijk_regress_dict[ijk] = r
            self.r2_score[tuple(ijk)] = r.score(x_poly, y)
            self.obs_to_var[tuple(ijk)] = obs_to_var

    def ijk_x_y_iter(self, rm_zero=True):
        """ iterates over each voxel

        Returns:
            ijk (tuple): ijk position
            x (np.array): x features
            y (np.array): y features
        """
        # get x for each sbj
        n_sbj = len(self.sbj_list)
        x_all = np.ones((n_sbj, len(self.x_label))) * np.nan
        for sbj_idx, sbj in enumerate(self.sbj_list):
            x_all[sbj_idx, :] = self.get_x_array(sbj)

        # get mask_all datacube (space0, space1, space2, n_sbj)
        mask_all = np.stack(
            (self._get_mask_array(sbj) for sbj in self.sbj_list),
            axis=3)

        # get datacube x (space0, space1, space2, len(self.x_label), n_sbj)
        y_all = np.stack([self.get_y_array(sbj) for sbj in self.sbj_list],
                         axis=4)

        # build mask_all which leaves only non zero vec
        zero_vec = np.zeros(len(self.y_label))

        for ijk, _ in np.ndenumerate(mask_all[:, :, :, 0]):
            # init
            x_list = list()
            y_list = list()
            mask = mask_all[ijk[0], ijk[1], ijk[2], :]

            # loop through by sbj, append x, y as needed
            for sbj_idx, (m, _x) in enumerate(zip(mask, x_all)):
                if not m:
                    # sbj mask is False
                    continue

                _y = y_all[ijk[0], ijk[1], ijk[2], :, sbj_idx]
                if rm_zero and np.array_equal(_y, zero_vec):
                    # dmri values are all zero
                    continue

                # add values
                x_list.append(_x)
                y_list.append(_y)

            if x_list:
                # only return if there are valid features @ ijk
                yield ijk, np.vstack(x_list), np.vstack(y_list)

    def get_y_array(self, sbj):
        """ loads data from nii files, returns array

        Args:
            sbj: key to self.sbj_img_tree

        Returns:
            y (np.array): (n_i, n_j, n_k, len(self.x_label)) features from img
        """
        y_list = list()
        for feat in self.y_label:
            img = nib.load(str(self.sbj_img_tree[sbj][feat]))
            y_list.append(img.get_data())
        return np.stack(y_list, axis=3)

    def get_x_array(self, sbj):
        """ gets features from sbj

        Args:
            sbj: key to self.sbj_img_tree

        Returns:
            x (np.array): (n_i, n_j, n_k, len(self.x_label)) features from img
        """
        x = np.ones(len(self.x_label)) * np.nan
        for x_idx, x_str in enumerate(self.x_label):
            x[x_idx] = getattr(sbj, x_str)

        if not np.isfinite(x).all():
            raise AttributeError('non finite y feature found')
        return x

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

    def r2_to_nii(self, f_out):
        img_r2 = nib.Nifti1Image(self.r2_score, self.ref_space.affine)
        img_r2.to_filename(str(f_out))

    def map_to(self):
        """ maps images to a set of y_feat """
        raise NotImplementedError
