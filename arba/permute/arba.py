from arba.plot import save_fig, size_v_pval
from arba.seg_graph import SegGraphHistory, SegGraphHistPval
from arba.space import Mask
from .permute import PermuteBase


class PermuteARBA(PermuteBase):
    """ runs ARBA permutations
    """
    flag_max = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sg_hist = None

    def save(self, folder, split, print_tree=False, **kwargs):
        """ saves output images in a folder"""
        super().save(folder=folder, split=split, **kwargs)

        if print_tree:
            assert self.sg_hist is not None, 'call determine_sig() first'

            merge_record = self.sg_hist.merge_record
            # todo: better way to store effect
            if self.file_tree.split_effect is not None:
                effect_mask = self.file_tree.split_effect[1].mask
            elif (folder / 'mask_effect.nii.gz').exists():
                effect_mask = Mask.from_nii(folder / 'mask_effect.nii.gz')
            else:
                effect_mask = None

            tree_hist, _ = merge_record.resolve_hist(self.file_tree, split)
            size_v_pval(tree_hist, mask=effect_mask,
                        mask_label='Effect Volume (%)')
            save_fig(f_out=folder / 'size_v_t2.pdf')

    def _split_to_sg_hist(self, split, pval_hist=False, **kwargs):
        """ builds sg_hist from a split

        Args:
            split (Split):
            pval_hist (bool): toggles whether sg_hist tracks history of pval

        Returns:
            sg_hist (SegGraphHistory): reduced as much as possible
        """
        if pval_hist:
            sg_hist = SegGraphHistPval(file_tree=self.file_tree, split=split,
                                       **kwargs)
        else:
            sg_hist = SegGraphHistory(file_tree=self.file_tree, split=split,
                                      **kwargs)
        return sg_hist

    def run_split(self, split, **kwargs):
        """ returns max stat (per vox) across new ARBA hierarchy

        Args:
            split (tuple): (num_sbj), split[i] describes which class the i-th
                           sbj belongs to in this splits

        Returns:
            sg_hist (SegGraphHistory): reduced as much as possible
        """
        sg_hist = self._split_to_sg_hist(split, **kwargs)
        sg_hist.reduce_to(1, **kwargs)

        return sg_hist

    def run_split_xtrm(self, split, **kwargs):
        """" returns the minimum pvalue across hierarchy
        """
        return split, self.run_split(split, **kwargs).min_pval

    def determine_sig(self, split=None, stat_volume=None):
        """ runs on the original case, uses the stats saved to determine sig"""

        # get volume of stat
        sg_hist = self.run_split(split, pval_hist=True)
        min_pval = sg_hist.get_min_pval_array()

        # store sg_hist of split
        self.sg_hist = sg_hist

        return super().determine_sig(stat_volume=min_pval)
