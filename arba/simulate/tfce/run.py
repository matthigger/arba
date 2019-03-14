import pathlib
import shlex
import shutil
import subprocess

import nibabel as nib


def run_tfce(f_data, nm, folder, num_perm=5000, f_mask=None, verbose=False, **kwargs):
    """ wrapper around randomise, interfaces at file level

    Args:
        f_data (str or Path): data cube (see build_data())
        nm (tuple): lengths of each group
        folder (str or Path): output of tfce
        num_perm (int): number of permutations
        f_mask (str or Path): mask

    Returns:
        f_tfce_pval (Path): file to nii of pval
    """

    f_design_ttest2 = shutil.which('design_ttest2')
    f_randomise = shutil.which('randomise')

    assert f_design_ttest2, 'design_ttest2 not found, FSL install / path?'
    assert f_randomise, 'randomise not found, FSL install / path?'

    # build folder
    folder = pathlib.Path(folder)
    folder.mkdir(exist_ok=True)

    # write design.mat, design.con
    f_design = folder / 'design'
    n, m = nm
    cmd = f'{f_design_ttest2} {f_design} {n} {m}'
    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    # call randomise
    cmd = f'randomise -i {f_data} -o {folder}/one_minus ' + \
          f'-d {f_design}.mat -t {f_design}.con ' + \
          f'-n {num_perm} -T'
    if not verbose:
        cmd += ' --quiet'
    if f_mask is not None:
        cmd += f' -m {f_mask}'
    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    # rewrite output pval' (fsl does 1-p ...)
    f_pval_list = list()
    for idx in [1, 2]:
        f_pval_flip = folder / f'one_minus_tfce_corrp_tstat{idx}.nii.gz'
        f_pval = folder / f'tfce_corrp_tstat{idx}_p.nii.gz'
        img_p_flip = nib.load(str(f_pval_flip))
        img_p = nib.Nifti1Image(1 - img_p_flip.get_data(), img_p_flip.affine)
        img_p.to_filename(str(f_pval))
        f_pval_list.append(f_pval)

    return f_pval_list