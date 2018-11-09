import nibabel as nib
import numpy as np

from .point_cloud import PointCloud
from .point_cloud_ijk import PointCloudIJK
from .ref_space import get_ref


class PointCloudXYZ(PointCloud):
    """ stores an array of points
    """

    def to_ijk(self, ref):
        """ maps into some ijk space (rounding to int), returns PointCloudIJK
        """

        ref = get_ref(ref)

        ijk = ref.from_rasmm(self.x, round_flag=True)

        return PointCloudIJK(ijk, ref)

    @staticmethod
    def from_nii(f_nii, **kwargs):
        """ read in nii mask
        """
        pc_ijk = PointCloudIJK.from_nii(f_nii, **kwargs)
        return pc_ijk.to_xyz()

    def to_nii(self, f_nii, ref, **kwargs):
        """ save nii mask
        """
        self.to_ijk(ref).to_nii(f_nii, **kwargs)

    @staticmethod
    def from_tract(f_trk):
        """ build from trk file
        """
        tract = nib.streamlines.load(str(f_trk))
        ref = get_ref(str(f_trk))

        if tract.streamlines:
            return sum(PointCloudXYZ(line, ref) for line in tract.streamlines)
        else:
            return PointCloudXYZ(np.zeros((0, 3)), ref)
