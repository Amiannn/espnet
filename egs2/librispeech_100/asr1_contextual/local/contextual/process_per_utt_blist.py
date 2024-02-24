import os

from local.rareword.utils import read_file
from local.rareword.utils import write_file

BLIST_PATH = "./local/rareword/all_rare_words.txt"

def get_uttblist(words, blist):
    return [word for word in words if word in blist]

if __name__ == '__main__':
    blist = [b[0] for b in read_file(BLIST_PATH, sp=' ')]
    print(blist[:10])

    datas_path = './dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test_clean' not in path:
            continue
        print(f'processing {path} ~')
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        for data in text_datas:
            uttid    = data[0]
            uttblist = get_uttblist(data[1:], blist)
            rareword_datas.append(
                [uttid] + (uttblist if len(uttblist) > 0 else [''])
            )

        output_path = os.path.join(path, 'uttblist')
        write_file(output_path, rareword_datas)