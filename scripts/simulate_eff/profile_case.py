from mh_pytools import file
from pnl_data.set.hcp_100 import folder

active_rad = 5
t2_list = [1]

folder_out = folder / 'profile'
# load
sim = file.load(folder_out / 'sim.p.gz')
sim.par_flag = False

# run arba
sim.run(t2_list, active_rad=active_rad)
