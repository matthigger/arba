import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict


def line_confidence(label_x_y_dict, shade_width=.95, ylabel='y', xlabel='x'):
    """

    Args:
        label_x_y_dict (dict): keys are (label, x), values are list of y
        shade_width (float): confidence range to shade
    """

    p_low = (1 - shade_width) * 100 / 2
    p_high = 100 - p_low

    # build color_dict
    label_set = {x[0] for x in label_x_y_dict.keys()}
    cm = plt.get_cmap('Set1')
    color_dict = {label: cm(idx) for idx, label in enumerate(sorted(label_set))}

    # build snr_dict_list, keys are labels, values are lists of tuples of
    # percentile data
    label_data_dict = defaultdict(list)
    for (label, x), y_list in label_x_y_dict.items():
        y_high = np.percentile(y_list, p_high)
        y_mean = np.mean(y_list)
        y_low = np.percentile(y_list, p_low)

        label_data_dict[label].append((x, y_high, y_mean, y_low))

    # plot
    sns.set(font_scale=1.2)
    for label, data in label_data_dict.items():
        # sort in increasing x
        data = sorted(data)
        x = [x[0] for x in data]
        y_high = [x[1] for x in data]
        y_mean = [x[2] for x in data]
        y_low = [x[3] for x in data]

        # plot
        plt.plot(x, y_mean, label=label, color=color_dict[label])
        plt.fill_between(x, y_high, y_low, color=color_dict[label], alpha=.2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)