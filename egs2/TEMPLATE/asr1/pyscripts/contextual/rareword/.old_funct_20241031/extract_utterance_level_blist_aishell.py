import os

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_UTT_BLIST = "./local/contextual/rarewords/train.bio.utt.json"
DEV_UTT_BLIST   = "./local/contextual/rarewords/dev.bio.utt.json"
TEST_UTT_BLIST  = "./local/contextual/rarewords/test.bio.utt.json"

TRAIN_DEV_BLIST_PATH = "./local/contextual/rarewords/rareword.train.sep.txt"
TEST_BLIST_PATH      = "./local/contextual/rarewords/rareword.all.sep.txt"

def get_uttblist(words, word2idx):
    return [[str(word2idx[word][0]), word2idx[word][1]] for word in words]

if __name__ == '__main__':
    datas_path = './dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path:
            blist_path = TRAIN_DEV_BLIST_PATH
        else:
            blist_path = TEST_BLIST_PATH
        if 'train' in path:
            utt_blist_list = read_json(TRAIN_UTT_BLIST)
        elif 'dev' in path:
            utt_blist_list = read_json(DEV_UTT_BLIST)
        elif 'test' in path:
            utt_blist_list = read_json(TEST_UTT_BLIST)

        blist = [b[0] for b in read_file(blist_path, sp=',')]
        word2idx = {word.replace(' ', ''): [i, word] for i, word in enumerate(blist)}
        print(word2idx)
        print(f'processing {path}...')
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        rareword_idxs  = []
        cache = {}
        for i, data in tqdm(enumerate(text_datas)):
            uttid       = data[0]
            uttid_no_sp = uttid.split('-')[-1]
            if uttid_no_sp in cache:
                uttbwords = cache[uttid_no_sp]
            else:
                uttbwords = utt_blist_list[i]
                cache[uttid_no_sp] = uttbwords
            print(uttbwords)
            results  = get_uttblist(uttbwords, word2idx)
            uttblist     = [d[1] for d in results]
            uttblist_idx = [d[0] for d in results]
            rareword_datas.append(
                [uttid] + (uttblist if len(uttblist) > 0 else [''])
            )
            rareword_idxs.append(
                [uttid] + (uttblist_idx if len(uttblist_idx) > 0 else [''])
            )

        output_path_uttblist = os.path.join(path, 'uttblist')
        write_file(output_path_uttblist, rareword_datas)

        output_path_uttblist_idx = os.path.join(path, 'uttblist_idx')
        write_file(output_path_uttblist_idx, rareword_idxs)