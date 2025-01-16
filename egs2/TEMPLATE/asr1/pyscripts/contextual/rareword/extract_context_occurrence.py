import os
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/contexts/context_keywords_test.txt"
TEST_BLIST_PATH      = "./local/contextual/contexts/context_f65536_train.txt"

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
    text_path  = './dump/raw/S95_sp/text'
    dev_path   = './dump/raw/dev/text'
    test_path  = './dump/raw/test/text'
    dump_path  = './local/contextual/contexts'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')
    name_test  = TEST_BLIST_PATH.split('/')[-1].replace('.txt', '')

    text_datas = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    dev_datas  = [[d[0], d[1:]] for d in read_file(dev_path, sp=' ')]
    test_datas = [[d[0], d[1:]] for d in read_file(test_path, sp=' ')]
    blist      = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=' ')]
    blist_test = [b[0] for b in read_file(TEST_BLIST_PATH, sp=' ')]
    
    # text_datas = text_datas + dev_datas
    counts      = list(map(lambda x: [str(x)], occurrence(text_datas, blist)))
    counts_test = list(map(lambda x: [str(x)], occurrence(test_datas, blist_test)))

    output_path = os.path.join(dump_path, f'{name}_occurrence_train.txt')
    write_file(output_path, counts)

    output_path = os.path.join(dump_path, f'{name_test}_occurrence_test.txt')
    write_file(output_path, counts_test)
