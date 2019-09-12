class Region:
    """ a volume and n, mu, cov of the observed features of each population

    this object serves as the nodes of a segmentation graph, they are
    aggregated greedily to minimize the 'error' function defined.

    Attributes:
        pc_ijk (PointCloud): set of voxels (tuples of ijk)
        fs_dict (dict): contains feat stats for img sets of different grps
    """

    def __init__(self, pc_ijk, fs_dict):
        self.pc_ijk = pc_ijk
        self.fs_dict = fs_dict

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

    @staticmethod
    def get_error(reg_1, reg_2, reg_u=None):
        raise NotImplementedError

    @staticmethod
    def from_data_img(data_img, ijk=None, pc_ijk=None):
        raise NotImplementedError
