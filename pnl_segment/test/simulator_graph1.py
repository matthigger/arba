import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mh_pytools import file
from pnl_data.set.cidar_post import folder

folder_out = folder / '2018_all_1'
f_out = folder_out / 'snr_auc.p.gz'
# save
method_snr_auc_dict = file.load(f_out)

method_set = set(method for method, _ in method_snr_auc_dict.keys())

# load
sns.set()
plt.subplots(1, 1)
plt.gca().set_xscale("log", nonposx='clip')
for method in method_set:
    snr_auc_dict = [(snr, np.mean(auc_list))
                    for (_method, snr), auc_list in method_snr_auc_dict.items()
                    if _method == method]
    snr_auc = sorted(snr_auc_dict)
    snr = [x[0] for x in snr_auc]
    auc = [x[1] for x in snr_auc]
    plt.plot(snr, auc, label=method)
plt.legend()
# plt.gca().set_xscale('log')
plt.gca().set_xlim(left=min(snr), right=max(snr))
plt.xlabel('snr')
plt.ylabel('auc')
plt.show()
