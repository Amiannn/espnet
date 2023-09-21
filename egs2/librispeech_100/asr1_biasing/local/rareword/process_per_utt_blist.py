import os

from local.rareword.utils import read_file
from local.rareword.utils import write_file

BLIST_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/local/rareword_f15.txt"

def get_uttblist(words, blist):
    return [word for word in words if word in blist]

if __name__ == '__main__':
    blist = [b[0] for b in read_file(BLIST_PATH, sp=' ')]
    print(blist[:10])

    datas_path = './dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        print(f'processing {path} ~')
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
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