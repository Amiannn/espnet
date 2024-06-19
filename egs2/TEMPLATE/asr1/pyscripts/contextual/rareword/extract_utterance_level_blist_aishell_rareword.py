import os

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/rarewords/rareword_f{2**1}_train.txt"
TEST_BLIST_PATH      = "./local/contextual/rarewords/rareword_f10_test.txt"

def get_uttblist(words, blist):
    return [[str(word2idx[word]), word] for word in words if word in blist]

if __name__ == '__main__':
    datas_path = './dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if 'zh' not in path:
            continue
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path:
            blist_path = TRAIN_DEV_BLIST_PATH
        else:
            blist_path = TEST_BLIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=' ')]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...')
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        rareword_idxs  = []
        for data in tqdm(text_datas):
            uttid    = data[0]
            results  = get_uttblist(data[1:], blist)
            uttblist     = [d[1] for d in results]
            uttblist_idx = [d[0] for d in results]
            rareword_datas.append(
                [uttid] + (uttblist if len(uttblist) > 0 else [''])
            )
            rareword_idxs.append(
                [uttid] + (uttblist_idx if len(uttblist_idx) > 0 else [''])
            )

        output_path_uttblist = os.path.join(path, f'uttblist')
        write_file(output_path_uttblist, rareword_datas)

        output_path_uttblist_idx = os.path.join(path, 'uttblist_idx')
        write_file(output_path_uttblist_idx, rareword_idxs)