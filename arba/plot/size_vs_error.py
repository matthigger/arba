import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def size_vs_error(sg_hist):
    size = list()
    error = list()

    for pg, _, _ in sg_hist:
        size.append(len(pg))
        error.append(sum(r.sq_error for r in pg.nodes))

    sns.set()
    plt.plot(size, error, label='error')

    max_err = max(error)
    max_size = max(size)
    error_reg = [e + (s - 1) * max_err / (max_size - 1) for e, s in
                 zip(error, size)]
    min_reg_size = min(zip(error_reg, size))[1]

    plt.plot(size, error_reg, label='regularized error')
    plt.axvline(min_reg_size, label=f'optimal size: {min_reg_size}', color='r')
    plt.legend()

    plt.xlabel('size (num regions in segmentation)')
    plt.ylabel('error (var added from voxel to region)')


def size_vs_error_normed(sg_hist, n_max=np.inf):
    size_error_dict = {len(pg): pg.sq_error for pg, _, _ in sg_hist}

    max_err = max(size_error_dict.values())

    # normalize and select only appropriate sizes
    size_error_dict = {s: e / max_err for s, e in size_error_dict.items()
                       if s <= n_max}

    # sns.set()
    size_error = sorted(size_error_dict.items())
    size = [x[0] for x in size_error]
    error = [x[1] for x in size_error]
    error_diff = [(e0 - e1) for e0, e1 in zip(error, error[1:])]
    error_diff_local = [(e0 - e1) / e1 for e0, e1 in zip(error, error[1:])]

    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.minorticks_on()
    plt.grid(True)
    plt.plot(size, error, label='error (normalized)')
    plt.plot(size[:-1], error_diff, label='diff error (normalized)')
    plt.plot(size[:-1], error_diff_local,
             label='error increase % (local normalized)')

    plt.legend()
    ax.tick_params(axis='y', which='minor', bottom=True)

    plt.xlabel('size (num reg in segmentation)')
    plt.ylabel('error')
