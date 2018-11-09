from pnl_segment.region.reg import Region


class RegionMinVar(Region):
    @property
    def var(self):
        return self._obj

    @property
    def error(self):
        return self._obj

    def get_obj(self):
        """ returns the weighted sum of |cov| in each grp
        """

        var_sum = [fs.cov_det * fs.n for fs in self.fs_dict.values()]

        # assume uniform prior of grps
        var_sum = sum(var_sum) / len(var_sum)

        return var_sum * len(self)
