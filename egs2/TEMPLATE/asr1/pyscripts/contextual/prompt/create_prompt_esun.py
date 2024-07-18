import os

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = "../asr1_contextual/local/contextual/rarewords/rareword_f1000_train.txt"
TEST_BLIST_PATH      = "../asr1_contextual/local/contextual/rarewords/rareword_f10_test.txt"

PROMPT_TEMPLATE         = '''今天的主題為: {}. 好! 我們開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''好! 我們開始吧.'''

def get_uttblist(words, blist):
    return [[str(word2idx[word]), word] for word in words if word in blist]

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
            
            if len(uttblist) > 0:
                prompt = PROMPT_TEMPLATE.format(", ".join(uttblist))
            else:
                prompt = PROMPT_NON_ENT_TEMPLATE
            rareword_datas.append([uttid, prompt.upper()])

        output_path_uttblist = os.path.join(path, 'prompt')
        write_file(output_path_uttblist, rareword_datas)