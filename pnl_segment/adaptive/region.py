import numpy as np


class Region:
    """ a set of voxels and their associated features

    Attributes:
        pc_ijk (PointCloudIJK)
        feat_stat (dict): contains feat stats for img sets of different grps
    """

    @property
    def obj(self):
        if self._obj is None:
            self._obj = self.get_obj()
        return self._obj

    def __init__(self, pc_ijk, feat_stat):
        self.pc_ijk = pc_ijk
        self.feat_stat = feat_stat
        self._obj = None

    def __str__(self):
        return f'{self.__class__} with {self.feat_stat}'

    def __add__(self, other):
        if isinstance(other, type(0)) and other == 0:
            # allows use of sum(reg_iter)
            return type(self)(self.pc_ijk, self.feat_stat)

        feat_stat = {grp: self.feat_stat[grp] + other.feat_stat[grp]
                     for grp in self.feat_stat.keys()}

        return type(self)(pc_ijk=self.pc_ijk + other.pc_ijk,
                          feat_stat=feat_stat)

    __radd__ = __add__

    def __len__(self):
        return len(self.pc_ijk)

    def __lt__(self, other):
        return len(self) < len(other)

    def get_obj(self):
        raise NotImplementedError('invalid in base class Region, see subclass')


class RegionMinVar(Region):
    def __init__(self, *args, grp_to_min_var=None, **kwargs):
        super().__init__(*args, **kwargs)

        if grp_to_min_var is None:
            self.grp_to_min_var = set(self.feat_stat.keys())
        else:
            self.grp_to_min_var = grp_to_min_var

    def get_obj(self):
        var_sum = 0
        for grp in self.grp_to_min_var:
            fs = self.feat_stat[grp]
            var_sum = np.linalg.det(fs.var) * fs.n

        # assume uniform prior of grps
        return var_sum / len(self.grp_to_min_var)

    @staticmethod
    def get_obj_pair(reg1, reg2):
        return (reg1 + reg2).obj - (reg1.obj + reg2.obj)
