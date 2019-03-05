import numpy as np

from mh_pytools import file, parallel
from pnl_data.set.hcp_100 import folder

folder = folder / '1000_FWER'
s_folder_glob = 't2*effect*'
arg_list = [{'folder': f} for f in folder.glob(s_folder_glob)]


def has_any_sig(folder):
    f_sg_arba_test_sig = folder / 'arba_cv' / 'save' / 'sg_arba_test_sig.p.gz'
    sg_arba_test_sig = file.load(f_sg_arba_test_sig)
    bool_sg_arba = bool(sg_arba_test_sig.nodes)

    # f_mask_sig_arba = folder / 'arba_cv' / 'mask_sig_arba_cv.nii.gz'
    # bool_mask = bool(nib.load(str(f_mask_sig_arba)).get_data().sum())
    # if bool_mask != bool_sg_arba:
    #     raise RuntimeError('mismatch')

    return bool_sg_arba


# has_any_sig(**arg_list[0])
res_out = parallel.run_par_fnc(has_any_sig, arg_list,
                               desc='reading per folder')

n = len(res_out)
n_success = sum(res_out)
n_fail = n - n_success
alpha_observed = n_success / n
conf = 1.96 / n * np.sqrt(n_success * n_fail / n)
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval
print(
    f'alpha_observed: {alpha_observed:.5f} +/- {conf} @ 95% using normal approx')
