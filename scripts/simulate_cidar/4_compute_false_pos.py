from mh_pytools import file, parallel
from pnl_data.set.cidar_post import folder
import numpy as np

folder = folder / '2019_Jan_02_14_54_46'
s_folder_glob = '*maha*run*'
arg_list = [{'folder': f} for f in folder.glob(s_folder_glob)]


def has_any_sig(folder):
    sg_arba_test_sig = file.load(folder / 'sg_arba_test_sig.p.gz')
    return bool(sg_arba_test_sig.nodes)


res_out = parallel.run_par_fnc(has_any_sig, arg_list,
                               desc='reading per folder')

n = len(res_out)
n_success = sum(res_out)
n_fail = n - n_success
alpha_observed = n_success / n
conf = 1.96 / n * np.sqrt(n_success * n_fail / n)
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval
print(f'alpha_observed: {alpha_observed:.5f} +/- {conf} @ 95% using normal approx')

