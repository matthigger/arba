from arba.simulate import tfce
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

alpha = .05
folder = folder / 'arba_cv_MF_FA-MD'
num_perm = 1000
verbose = True
par_flag = True

ft_dict = file.load(folder / 'save' / 'ft_dict_.p.gz')

folder_tfce = folder / 'tfce'
folder_tfce.mkdir(exist_ok=True)
tfce.compute_tfce(ft_dict, alpha=alpha, folder=folder_tfce, num_perm=num_perm,
                  verbose=verbose, par_flag=par_flag)
