from arba.simulate import tfce
from mh_pytools import file
from pnl_data.set.sz import folder

alpha = .05
folder = folder / 'arba_cv_HC-FES_fa-md'
num_perm = 1000
verbose = True

ft_dict = file.load(folder / 'save' / 'ft_dict_.p.gz')
ft_tuple = tuple(ft_dict.values())

folder_tfce = folder / 'tfce'
folder_tfce.mkdir(exist_ok=True)
tfce.compute_tfce(ft_tuple, alpha=alpha, folder=folder_tfce, num_perm=num_perm,
                  verbose=verbose)
