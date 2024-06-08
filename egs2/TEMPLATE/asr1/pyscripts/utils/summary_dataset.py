import os
import json

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

if __name__ == '__main__':
    dump_folder_paths = [
        './dump/raw/test',
        './dump/raw/dev',
        './dump/raw/train_sp',
    ]

    for dump_folder_path in dump_folder_paths:
        wav_scp_path   = os.path.join(dump_folder_path, 'utt2num_samples')
        datas = [int(d[1]) for d in read_file(wav_scp_path, sp=' ')]
        hours = (sum(datas) / 16000) / 3600
        print(f'{dump_folder_path.split("/")[-1]}: {hours:.2f}')

        non_sp_datas = [int(d[1]) for d in read_file(wav_scp_path, sp=' ') if 'sp' not in d[0]]
        hours        = (sum(non_sp_datas) / 16000) / 3600
        print(f'Non-sp {dump_folder_path.split("/")[-1]}: {hours:.2f}')
        print(f'_' * 30)
