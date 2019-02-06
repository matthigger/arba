from arba.region.reg import Region


class RegionWard(Region):
    """ Ward clustering (minimize determinant of covariance)

    todo: weights per grp
    todo: weights per feature
    """

    def __init__(self):
        raise NotImplementedError('not tested')

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        if reg_u is None:
            reg_u = reg_1 + reg_2

        return len(reg_u) * reg_u.cov_det - \
               len(reg_1) * reg_1.cov_det - \
               len(reg_2) * reg_2.cov_det

    @property
    def cov_det(self):
        """ returns the weighted sum of |cov| in all grp (equal weight per grp)
        """

        return sum(fs.cov_det for fs in self.fs_dict.values())
