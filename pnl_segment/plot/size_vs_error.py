import matplotlib.pyplot as plt
import seaborn as sns


def size_vs_error(sg_hist):
    size = list()
    error = list()

    for pg in sg_hist:
        size.append(len(pg))
        error.append(sum(r.error for r in pg.nodes))

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


def size_vs_error_normed(sg_hist):
    size = list()
    error = list()

    for pg in sg_hist:
        size.append(len(pg))
        error.append(sum(r.error for r in pg.nodes))

    sns.set()
    #plt.plot(size, error, label='error')

    max_err = max(error)
    error_reg = [e / max_err for e in error]
    error_reg_perc = [(e1 - e0) / e0 for e0, e1 in zip(error, error[1:])]
    plt.plot(size, error_reg, label='normalized error')
    plt.plot(range(len(error_reg), 1, -1), error_reg_perc, label='delta error %')
    plt.legend()
    plt.gca().set_xscale('log')
    # plt.gca().set_yscale('log')

    plt.xlabel('size (num regions in segmentation)')
    plt.ylabel('normalized error (var added from voxel to region)')
