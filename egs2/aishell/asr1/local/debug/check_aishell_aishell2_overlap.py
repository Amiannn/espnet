import os

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import write_file
from pyscripts.contextual.utils.dataio import write_json

AISHELL_1_TEST_PATH  = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/data/test/text"
AISHELL_1_TRAIN_PATH = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/data/train/text"
AISHELL_2_TRAIN_PATH = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell2/asr1/data/train/text"

DUMP_PATH = "./local/debug"

if __name__ == '__main__':
    aishell_1_test_datas  = read_file(AISHELL_1_TEST_PATH, sp=' ')
    aishell_1_train_datas = read_file(AISHELL_1_TRAIN_PATH, sp=' ')
    aishell_2_train_datas = [d[1] for d in read_file(AISHELL_2_TRAIN_PATH, sp='\t')]

    datas = {
        'ai1_train': aishell_1_train_datas,
        'ai1_test' : aishell_1_test_datas,
    }

    hits = {name: [] for name in datas}
    for dataset_name in datas:
        print(f'process {dataset_name}...')
        for uid, a1 in tqdm(datas[dataset_name]):
            if a1 in aishell_2_train_datas:
                hits[dataset_name].append([uid, a1])
        print(f'{dataset_name} hits: {len(hits[dataset_name])}')
        output_path = os.path.join(DUMP_PATH, f'ai2_{dataset_name}_overlap.json')
        write_json(output_path, hits[dataset_name])