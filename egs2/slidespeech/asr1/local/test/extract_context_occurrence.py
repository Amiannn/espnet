import os
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/contexts/context_keywords_test.txt"
TEST_BLIST_PATH      = "./local/contextual/contexts/context_f10_test.txt"

def occurrence(texts, bwords):
    domains = set([d[0].split('_')[0] for d in texts])

    bword_occurrence = {name: {word: 0 for word in bwords} for name in domains}
    oov = 0
    for uid, words in texts:
        name = uid.split('_')[0]
        for word in words:
            if word not in bword_occurrence[name]:
                oov += 1
                continue
            bword_occurrence[name][word] += 1
    return {name: list(bword_occurrence[name].values()) for name in domains}

if __name__ == '__main__':
    text_path  = './dump/raw/test/text'
    dev_path   = './dump/raw/dev/text'
    dump_path  = './local/contextual/contexts'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')

    text_datas = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    dev_datas  = [[d[0], d[1:]] for d in read_file(dev_path, sp=' ')]
    blist      = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=' ')]
    
    # text_datas = text_datas + dev_datas
    result = occurrence(text_datas, blist)
    for domain in result:
        counts = list(map(lambda x: [str(x)], result[domain]))
        output_path = os.path.join(dump_path, f'{name}_{domain}_occurrence.txt')
        write_file(output_path, counts)
