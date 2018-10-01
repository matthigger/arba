import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
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
        # get x for each sbj (num_sbj, len(self.x_label))
        x_all = self.get_x()

        # get y (num_i, num_j, num_k, len(self.y_label), num_sbj)
        y_all = self.get_y()

        # get mask_all datacube (num_i, num_j, num_k, num_sbj)
        mask_all = self.get_mask()

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

    def get_x(self, sbj_list=None):
        """ gets features from all sbj

        Args:
            sbj_list: list of sbj

        Returns:
            x (np.array): (num_sbj, len(self.x_label)) sbj features
        """
        if sbj_list is None:
            sbj_list = self.sbj_list

        x = np.ones((len(sbj_list), len(self.x_label))) * np.nan
        for sbj_idx, sbj in enumerate(sbj_list):
            for x_idx, x_str in enumerate(self.x_label):
                x[sbj_idx, x_idx] = getattr(sbj, x_str)

        if not np.isfinite(x).all():
            raise AttributeError('non finite y feature found')
        return x

    def get_y(self, sbj_list=None):
        """ loads data from nii files, returns array

        Args:
            sbj_list: list of sbj

        Returns:
            y (np.array): (num_i, num_j, num_k, num_x, num_sbj) feat from img
        """
        if sbj_list is None:
            sbj_list = self.sbj_list

        y_out = list()
        for sbj in sbj_list:
            sbj_y_list = list()
            for feat in self.y_label:
                img = nib.load(str(self.sbj_img_tree[sbj][feat]))
                sbj_y_list.append(img.get_data())
            y_out.append(np.stack(sbj_y_list, axis=3))
        return np.stack(y_out, axis=4)

    def _get_mask_per_sbj(self, sbj):
        try:
            # load and return mask
            f = self.sbj_mask_dict[sbj]
            img = nib.load(str(f))
            return img.get_data()
        except KeyError:
            # no mask given for sbj
            return np.ones(shape=self.ref_space.shape)

    def get_mask(self, sbj_list=None):
        """ gets mask (builds array of ones if sbj not in self.sbj_mask_dict)

        Args:
            sbj_list (list): list of sbj

        Returns:
            mask (np.array): mask (num_i, num_j, num_k, len(sbj_list))
        """
        if sbj_list is None:
            sbj_list = self.sbj_list

        mask_list = [self._get_mask_per_sbj(sbj) for sbj in sbj_list]
        return np.stack(mask_list, axis=3)

    def r2_to_nii(self, f_out):
        img_r2 = nib.Nifti1Image(self.r2_score, self.ref_space.affine)
        img_r2.to_filename(str(f_out))

    def plot(self, ijk, x_feat, y_feat, n_pts=100):
        """ plots observations and trend line

        Args:
            ijk (tuple): which voxel to examine
            x_feat: element of self.x_label to examine
            y_feat: element of self.y_label to examine
        """
        r = self.ijk_regress_dict[tuple(ijk)]

        # get data
        x_obs = self.get_x()
        y_obs = self.get_y()[ijk[0], ijk[1], ijk[2], :, :]

        # get idx
        x_feat_idx = self.x_label.index(x_feat)
        y_feat_idx = self.y_label.index(y_feat)

        def get_linspace_n_std_dev(z, n_std=3):
            """ returns equally spaced vector in mu +/- std * n_std to mu
            """
            mu = np.mean(z)
            delta = np.std(z) * n_std
            return np.linspace(mu - delta, mu + delta, n_pts)

        # compute prediction
        x_domain = np.vstack([np.mean(x_obs, axis=0)] * n_pts)
        x_domain[:, x_feat_idx] = get_linspace_n_std_dev(x_obs[:, x_feat_idx])
        y_est = r.predict(self.poly.fit_transform(x_domain))

        # todo: change indexing so sbj always @ end
        sns.set()
        plt.scatter(x_obs[:, x_feat_idx],
                    y_obs[y_feat_idx, :], label='observed')

        # todo: dimensions?
        plt.plot(x_domain[:, x_feat_idx],
                 y_est[:, y_feat_idx], label='estimated')

        # label
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.legend()

    def map_to_sbj(self, sbj_to, folder_out, sbj_from_list=None, verbose=True):
        """ maps images to demographics given in sbj_to and saves nii
        """
        if not len(self.ijk_regress_dict):
            raise AttributeError('call fit() before mapping')

        if sbj_from_list is None:
            sbj_from_list = self.sbj_list

        # get input to fnc
        def get_x_poly(sbj):
            """ returns polynom input to obj in self.ijk_regress_dict.values()

            we compute this once rather than call predict for comp efficiency
            """
            return self.poly.fit_transform(self.get_x([sbj]))

        x_to = get_x_poly(sbj_to)

        tqdm_dict = {'disable': not verbose, 'desc': f'to {sbj_to} per sbj'}
        for sbj_from in tqdm(sbj_from_list, **tqdm_dict):
            # get sbj from
            x_from = get_x_poly(sbj_from)

            # load original data
            y = self.get_y([sbj_from])[:, :, :, :, 0]

            # adjust for demographics
            for ijk in self.ijk_regress_dict.keys():
                r = self.ijk_regress_dict[ijk]
                delta = r.predict(x_to) - r.predict(x_from)
                y[ijk[0], ijk[1], ijk[2], :] += np.squeeze(delta)

            # print to nii
            for y_idx, y_feat in enumerate(self.y_label):
                # build output filename
                f_orig = self.sbj_img_tree[sbj_from][y_feat]
                f_out = folder_out / f_orig.name
                if f_out == f_orig:
                    raise FileExistsError(f'identical orig file: {f_out}')

                # write output
                img = nib.Nifti1Image(y[:, :, :, y_idx], self.ref_space.affine)
                img.to_filename(str(f_out))

    def predict(self, ijk, sbj):
        x = self.poly.fit_transform(self.get_x([sbj]))
        return self.ijk_regress_dict[ijk].predict(x)
