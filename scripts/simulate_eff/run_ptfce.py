import shutil

from tqdm import tqdm

from arba.simulate import PermutePTFCE
from mh_pytools import file
from pnl_data.set.hcp_100 import folder

folder_exp = folder / 'result' / 'FAMD_n70_size250_nperm1000_cube_full'
num_perm = 300

for folder_eff in tqdm(folder_exp.glob('t2_*'), total=70):
    permute_old = file.load(folder_eff / 'tfce' / 'permute.p.gz')
    split = permute_old.file_tree.split_effect[0]
    file_tree = permute_old.file_tree
    del permute_old

    _folder = folder_eff / 'ptfce'
    if _folder.exists():
        shutil.rmtree(str(_folder))
    _folder.mkdir(exist_ok=True)

    permute = PermutePTFCE(file_tree)
    permute.run(split, n=num_perm, folder=_folder, par_flag=True, verbose=True,
                save_self=True, print_hist=True)

    print(f'\n\ndone: {_folder}')