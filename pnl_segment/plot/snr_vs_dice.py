from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from ..region.stats import BenHochYek


def snr_vs_dice(sg_eff_dict, specificity=.05):
    """

    Args:
        sg_eff_dict (dict): keys are label, values are seg_graph, effect
        specificity (float): sig threshold
    """

    # compute dice per seg_graph
    snr_dice_dict = defaultdict(list)
    for label, sg_eff_list in sg_eff_dict.items():
        for (sg, eff) in sg_eff_list:
            # determines if significant
            # thresh = specificity / len(sg)
            p_list = [reg.pval for reg in sg.nodes]
            thresh = BenHochYek(p_list, sig=specificity)

            def is_sig(reg):
                return reg.pval < thresh

            # get binary array per label
            x = sg.to_array(is_sig)

            # compute dice
            dice = eff.get_dice(x)

            snr_dice_dict[label].append((eff.snr, dice))

    # plot
    sns.set()
    fig, ax = plt.subplots(1, 1)
    for label, snr_dice_list in snr_dice_dict.items():
        snr_dice_list = sorted(snr_dice_list)
        snr_list = [x[0] for x in snr_dice_list]
        dice_list = [x[1] for x in snr_dice_list]

        plt.plot(snr_list, dice_list, label=label)
    plt.legend()
    plt.xlabel('snr')
    plt.ylabel('dice (1=perfect)')
