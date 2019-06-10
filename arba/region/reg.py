import arba.bayes


class Region:
    """ a volume and n, mu, cov of the observed features of all populations

    this object serves as the nodes of a segmentation graph, they are
    aggregated greedily to minimize the 'error' function defined.

    Attributes:
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        fs_dict (dict): contains feat stats for img sets of different grps
    """

    def __init__(self, pc_ijk, fs_dict):
        self.pc_ijk = pc_ijk
        self.fs_dict = fs_dict

    def from_ft_dict(self, ft_dict, pc_ijk=None):
        if pc_ijk is None:
            pc_ijk = self.pc_ijk

        r_list = list()
        for ijk in pc_ijk:
            fs_dict = {grp: ft_dict[grp].ijk_fs_dict[ijk]
                       for grp in self.fs_dict.keys()}
            r = type(self)(pc_ijk={ijk}, fs_dict=fs_dict)
            r_list.append(r)

        if not r_list:
            raise RuntimeError()

        return sum(r_list)

    def __str__(self):
        return f'{self.__class__} @ {self.pc_ijk} w/ {self.fs_dict}'

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.fs_dict)

        if not isinstance(other, type(self)):
            raise TypeError

        fs_dict = {grp: self.fs_dict[grp] + other.fs_dict[grp]
                   for grp in self.fs_dict.keys()}

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict)

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)

    def bayes_mu(self, num_vox_prior_weight=10, **kwargs):
        """ computes bayes """
        fs_sum = sum(self.fs_dict.values())
        # prior has same weight as a single voxel's observation
        num_obs_prior = int(fs_sum.n / len(self)) * num_vox_prior_weight

        grp_mu_cov_dict = dict()
        for grp, fs in self.fs_dict.items():
            mvnorm = arba.bayes.MVNorm.non_inform_dplus1(len(fs_sum.mu),
                                                         mu=fs_sum.mu,
                                                         num_obs=num_obs_prior,
                                                         cov=fs_sum.cov)
            mvnorm = mvnorm.bayes_update(obs_mu=fs.mu,
                                         obs_cov=fs.cov,
                                         num_obs=fs.n)
            deg_free, loc, shape = mvnorm.get_mu_marginal()

            # assume params of multivariate t are gaussian (converges as
            # deg_free diverges)
            grp_mu_cov_dict[grp] = loc, shape

        return grp_mu_cov_dict

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        raise NotImplementedError
