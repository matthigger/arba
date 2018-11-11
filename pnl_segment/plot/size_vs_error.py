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
    plt.ylabel('error in cluster (var added from voxel to region)')
