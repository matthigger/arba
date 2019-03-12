import multiprocessing
import os
import shlex
import subprocess
import tempfile
from bisect import bisect_right

import nibabel as nib
import numpy as np
from tqdm import tqdm

from mh_pytools import file
from mh_pytools.parallel import run_par_fnc
from ...region import FeatStat
from ...space import PointCloud


def get_max_tfce_t2(folder, f_out='max_tfce_t2.p.gz', delete_glob='temp*',
                    max_tfce_t2=None):
    """ aggregates and grooms all max_tfce_t2 files in a folder """
    # init as empty if none passed
    if max_tfce_t2 is None:
        max_tfce_t2 = dict()

    # update with saved versions
    for f in folder.glob('*max_tfce_t2*.p.gz'):
        _max_tfce_t2 = file.load(f)
        split_new = set(_max_tfce_t2.keys())
        split_old = set(max_tfce_t2.keys())
        for split in (split_new & split_old):
            assert np.isclose(_max_tfce_t2[split],
                              max_tfce_t2[split]), 'one split, two max_tfce_t2'

        # update
        max_tfce_t2.update(_max_tfce_t2)

    print(f'{len(max_tfce_t2)} splits found')

    # save into one place
    file.save(max_tfce_t2, folder / f_out)

    # delete all old ones
    for f in folder.glob(delete_glob):
        os.remove(str(f))

    return max_tfce_t2


def permute_tfce(x, mask, split, n=5000, par_flag=False, seed=1, folder=None,
                 affine=None, verbose=False, additive=False):
    """ runs permutation on tfce enhanced images

    (NOTE: FSL has this functionality, but not parallel unless you have
    sun grid engine)

    Args:
        x (np.array): (shape0, shape1, shape2, num_sbj, num_feat)
                            features
        mask (np.array): (shape0, shape1, shape2) only computed where mask
                 is true
        split (np.array): (num_sbj) which group each subject belongs to, values
                          must be either 0 or 1
        n (int): number of permutations to test
        par_flag (bool): toggles parallel computation
        seed : init random seed (to determine partitions)
        folder (str or Path): if passed, outputs files to this location
        affine (np.array): affine of output images (if folder passed)
        verbose (bool): toggles command line output
        additive (bool): toggles 'additive' mode, in this mode, we see if any
                         splits were previously run in the folder.  saves
                         computation time if so (and hedges against bugs!)

    Returns:
        tfce_t2 (np.array): t2 image of tfce enhanced split
        max_tfce_t2 (dict): keys are split tuples, values are  max t2 of tfce
                            enhanced image
        pval (np.array): pvalue of each

    """
    if additive:
        # load max stats which have already been computed
        max_tfce_t2 = get_max_tfce_t2(folder)

        # reduce n by the already computed amount
        n = n - len(max_tfce_t2)

        if n <= 0:
            raise RuntimeError('already computed for n as large, quitting')
    else:
        max_tfce_t2 = dict()

    # sample split_set (a split may be repeated)
    np.random.seed(seed)
    num_sbj = x.shape[3]
    split_list = list()
    num_ones = int(split.sum())
    while len(split_list) < n:
        # each permuted split has same number of 1s as the split passed
        ones_idx = np.random.choice(range(num_sbj), size=num_ones,
                                    replace=False)
        split = tuple(idx in ones_idx for idx in range(num_sbj))
        if split not in max_tfce_t2.keys():
            split_list.append(split)

    # compute max tfce t2 per split
    if par_flag:
        # build arg list
        arg_list = list()
        n_cpu = min(multiprocessing.cpu_count(), n)
        idx = np.linspace(0, n, n_cpu + 1).astype(int)
        for cpu_idx in range(n_cpu):
            _split_list = split_list[idx[cpu_idx]:idx[cpu_idx + 1]]
            d = {'x': x,
                 'split_list': _split_list,
                 'mask': mask,
                 'verbose': verbose and (cpu_idx == n_cpu - 1)}
            if folder is not None:
                d['f_out'] = folder / f'temp_max_tfce_t2_{cpu_idx}.p.gz'
            arg_list.append(d)

        # run
        res = run_par_fnc(get_max_t2, arg_list, verbose=verbose)

        # aggregate + sort
        for _max_tfce_t2 in res:
            max_tfce_t2.update(_max_tfce_t2)
    else:
        f_out = folder / f'temp_max_tfce_t2.p.gz'
        _max_tfce_t2 = get_max_t2(x, split_list, mask, verbose=verbose,
                                  f_out=f_out)
        max_tfce_t2.update(_max_tfce_t2)

    # build t2 image of x given original split
    pc = PointCloud.from_mask(mask)
    tfce_t2 = get_t2(x, split, pc=pc, verbose=verbose)
    tfce_t2 = apply_tfce(tfce_t2)

    # build array of pval and conf
    # https://stats.stackexchange.com/questions/109207/p-values-equal-to-0-in-permutation-test
    max_tfce_t2_sorted = sorted(max_tfce_t2.values())
    pval = np.zeros(x.shape[:3])
    for i, j, k in pc:
        p = bisect_right(max_tfce_t2_sorted, tfce_t2[i, j, k]) / n
        pval[i, j, k] = 1 - p

    if folder is not None:
        # save
        get_max_tfce_t2(folder=folder, max_tfce_t2=max_tfce_t2)

        if affine is None:
            print('warning: identity affine used, pass affine to avoid this')
            affine = np.eye(4)

        file.save(max_tfce_t2, folder / 'max_tfce_t2.p.gz')

        img_tfce_t2 = nib.Nifti1Image(tfce_t2, affine=affine)
        img_tfce_t2.to_filename(str(folder / 'tfce_t2.nii.gz'))

        img_pval = nib.Nifti1Image(pval, affine=affine)
        img_pval.to_filename(str(folder / 'pval.nii.gz'))

    return tfce_t2, max_tfce_t2, pval


def get_t2(x, split, pc=None, verbose=False):
    split = np.array(split)

    # build fs per ijk in mask
    t2 = np.zeros(pc.ref.shape)
    tqdm_dict = {'disable': not verbose,
                 'desc': 'compute t2 per vox'}
    for i, j, k in tqdm(pc, **tqdm_dict):
        # get feat stat of each grp
        fs0 = FeatStat.from_array(x[i, j, k, split == 0, :].T, _fast=True)
        fs1 = FeatStat.from_array(x[i, j, k, split == 1, :].T, _fast=True)

        # compute t2
        delta = fs0.mu - fs1.mu
        cov_pooled = (fs0.cov * fs0.n + fs1.cov * fs1.cov) / (fs0.n + fs1.n)
        t2[i, j, k] = delta @ np.linalg.inv(cov_pooled) @ delta

    return t2


def to_file(x, tag=''):
    f = tempfile.NamedTemporaryFile(suffix=f'{tag}.nii.gz').name
    img_mask = nib.Nifti1Image(x, affine=np.eye(4))
    img_mask.to_filename(f)
    return f


def apply_tfce(x, **kwargs):
    """ applies tfce to an array, deletes files """

    # get input / output files
    f_x = to_file(x)
    f_out = tempfile.NamedTemporaryFile(suffix='_tfce.nii.gz').name

    # compute
    apply_tfce_file(f_in=f_x, f_out=f_out, **kwargs)

    # cleanup
    x_tfce = nib.load(f_out).get_data()
    os.remove(f_out)
    os.remove(f_x)

    return x_tfce


def apply_tfce_file(f_in, f_out=None, h=2, e=.5, c=6):
    # get f_out
    if f_out is None:
        f_out = tempfile.NamedTemporaryFile(suffix='_tfce.nii.gz').name
    # call randomise
    cmd = f'fslmaths {f_in} -tfce {h} {e} {c} {f_out}'

    p = subprocess.Popen(shlex.split(cmd))
    p.wait()

    return f_out


def get_max_t2(x, split_list, mask, verbose=False, f_out=None):
    """ per split_tuple: get t2, tfce enhance, find max value (all within mask)

    Args:
        mask (np.array): (shape0, shape1, shape2) only computed where mask
                         is true
        x (np.array): (shape0, shape1, shape2, num_sbj, num_feat)
                      features
        split_list (list): list of tuples, each tuple is of length num_sbj and
                           has boolean values
        verbose (bool): toggles command line output
        f_out (Path): if passed, saves max_tfce_t2 while running

    Returns:
        max_tfce_t2 (dict): keys are split tuples, values are  max t2 of tfce
                            enhanced image
    """
    # compute point cloud of mask (set of ijk idx)
    pc = PointCloud.from_mask(mask)

    max_tfce_t2 = dict()
    tqdm_dict = {'disable': not verbose,
                 'desc': 'compute t2, apply tfce, take max per split'}
    for split_tuple in tqdm(split_list, **tqdm_dict):
        # compute t2
        t2 = get_t2(x, split_tuple, pc=pc, verbose=verbose)

        # apply tfce
        t2_tfce = apply_tfce(x=t2)

        # get max value within mask
        max_tfce_t2[tuple(split_tuple)] = t2_tfce[mask].max()

        if f_out is not None:
            file.save(max_tfce_t2, f_out)

    return max_tfce_t2
