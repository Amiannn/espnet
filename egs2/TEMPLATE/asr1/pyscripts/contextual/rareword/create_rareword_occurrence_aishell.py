import os
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/rarewords/rareword_f{2**1}_train.txt"
TEST_BLIST_PATH      = "./local/contextual/rarewords/rareword_f10_test.txt"

def occurrence(texts, bwords):
    bword_occurrence = {word: 0 for word in bwords}
    oov = 0
    for uid, words in texts:
        for word in words:
            if word not in bword_occurrence:
                oov += 1
                continue
            bword_occurrence[word] += 1
    return list(bword_occurrence.values()) + [oov]

if __name__ == '__main__':
    text_path  = './dump/raw/zh_train_sp/text'
    dev_path   = './dump/raw/zh_dev/text'
    dump_path  = './local/contextual/rarewords'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')

    text_datas = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    dev_datas  = [[d[0], d[1:]] for d in read_file(dev_path, sp=' ')]
    blist      = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=' ')]
    
    text_datas = text_datas + dev_datas
    counts     = list(map(lambda x: [str(x)], occurrence(text_datas, blist)))

    output_path = os.path.join(dump_path, f'{name}_occurrence.txt')
    write_file(output_path, counts)
