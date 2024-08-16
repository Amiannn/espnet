import os

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

root_path    = './dump/raw'
folder_names = ['train_sp', 'dev', 'test']

for folder_name in folder_names:
    path = os.path.join(root_path, folder_name)
    print(f'path: {path}')

    # text
    text_path  = os.path.join(path, 'text')
    text_datas = [[d[0], " ".join(d[1:])] for d in read_file(text_path, sp=' ')]

    keep_uid        = []
    keep_text_datas = []
    length_couts = {i: 0 for i in range(500)}
    for uid, words in text_datas:
        length = len(words.split(" "))
        length_couts[length] += 1
        if len(words.split(" ")) <= 32:
            keep_uid.append(uid)
            keep_text_datas.append([uid, words])
    output_text_path = f'{text_path}_'
    write_file(output_text_path, keep_text_datas, sp=" ")
    print(f'total {len(text_datas)}, keep {len(text_datas) - len(keep_uid)}')

    output_text_path = os.path.join(path, 'lengths')
    write_file(output_text_path, [[str(v)] for v in list(length_couts.values())], sp=" ")

    # wav.scp
    wav_path  = os.path.join(path, 'wav.scp')
    wav_datas = read_file(wav_path, sp=' ')

    keep_wav_datas = []
    for uid, wpath in wav_datas:
        if uid in keep_uid:
            keep_wav_datas.append([uid, wpath])
    output_wav_path = f'{wav_path}_'
    write_file(output_wav_path, keep_wav_datas, sp=" ")

    # uttbilst
    uttblist_path  = os.path.join(path, 'uttblist_idx')
    uttblist_datas = [[d[0], " ".join(d[1:])] for d in read_file(uttblist_path, sp=' ')]
    keep_uttb_datas = []
    for uid, uttb in uttblist_datas:
        if uid in keep_uid:
            keep_uttb_datas.append([uid, uttb])
    output_uttb_path = f'{uttblist_path}_'
    write_file(output_uttb_path, keep_uttb_datas, sp=" ")