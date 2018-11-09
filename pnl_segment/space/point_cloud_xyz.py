import nibabel as nib
import numpy as np

from . import point_cloud_ijk, point_cloud, ref_space


class PointCloudXYZ(point_cloud.PointCloud):
    """ stores an array of points
    """

    def to_ijk(self, ref):
        """ maps into some ijk space (rounding to int), returns PointCloudIJK
        """

        ref = ref_space.get_ref(ref)

        ijk = ref.from_rasmm(self.x, round_flag=True)

        return point_cloud_ijk.PointCloudIJK(ijk, ref)

    @staticmethod
    def from_nii(f_nii, **kwargs):
        """ read in nii mask
        """
        pc_ijk = point_cloud_ijk.PointCloudIJK.from_nii(f_nii, **kwargs)
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
        ref = ref_space.get_ref(str(f_trk))

        if tract.streamlines:
            return sum(PointCloudXYZ(line, ref) for line in tract.streamlines)
        else:
            return PointCloudXYZ(np.zeros((0, 3)), ref)
