import os

from local.utils import read_file
from local.utils import write_file

folders   = ['train_sp', 'dev', 'test']
root_path = "./dump/raw"

ws_path   = '/share/nas165/amian/experiments/speech/preprocess/dataset_pre/egs2/esun_investor_various/tsv_ws/all.tsv'

if __name__ == '__main__':
    ws_datas = {d[0]: d[2] for d in read_file(ws_path, sp='\t')}

    for folder in folders[:1]:
        path  = os.path.join(root_path, folder, 'text')
        datas = [[d[0], " ".join(d[1:])] for d in read_file(path, sp=' ')]

        for id, text in datas:
            ws_text = ws_datas[id]
            print(f'text   : {text}')
            print(f'ws text: {ws_text}')
            print('_' * 30)