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
        self.__obj = None
        self._pval = None
        self._pval_override = None

    def __str__(self):
        return f'{self.__class__} @ {self.pc_ijk} w/ {self.fs_dict}'

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.fs_dict)

        fs_dict = {grp: self.fs_dict[grp] + other.fs_dict[grp]
                   for grp in self.fs_dict.keys()}

        return type(self)(pc_ijk=self.pc_ijk | other.pc_ijk,
                          fs_dict=fs_dict)

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)

    @property
    def error(self):
        raise NotImplementedError

    @property
    def pval(self):
        if self._pval_override is not None:
            return self._pval_override

        if self._pval is None:
            self._pval = self.get_pval()
        return self._pval

    @pval.setter
    def pval(self, pval):
        # if pval is set, it is set permanently
        self._pval_override = pval

    @staticmethod
    def get_error_delta(reg_1, reg_2, reg_union=None):
        if reg_union is None:
            reg_union = reg_1 + reg_2
        return reg_union.error - reg_1.error - reg_2.error

    @property
    def _obj(self):
        # memoize
        if self.__obj is None:
            self.__obj = self.get_obj()
        return self.__obj

    def get_pval(self):
        raise NotImplementedError

    def get_obj(self):
        raise NotImplementedError

    def reset(self):
        self.__obj = None
        self._pval = None
