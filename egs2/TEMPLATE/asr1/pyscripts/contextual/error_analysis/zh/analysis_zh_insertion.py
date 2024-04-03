import os
import json

from jiwer import cer
from tqdm  import tqdm

from rapidfuzz import fuzz
from pypinyin  import lazy_pinyin


from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.contextual.utils.aligner import CheatDetector
from pyscripts.contextual.utils.aligner import align_to_index

def encode(word):
    return " ".join(lazy_pinyin(word))

def similarity(query, value):
    score = fuzz.ratio(query, value) / 100
    return score

if __name__ == '__main__':
    
    datas = read_json('./aishell_insertion.json')
    char_scores = 0
    pin_scores  = 0
    count       = 0
    for uid in datas:
        data = datas[uid]
        for rchar, hchar in data:
            rpin = encode(rchar)
            hpin = encode(hchar)
            char_score = similarity(hchar, rchar)
            pin_score  = similarity(hpin, rpin)
            char_scores += char_score
            pin_scores  += pin_score
            count += 1
    
    final_char_score = char_scores / count
    final_pin_score  = pin_scores / count

    print(f'final_char_score: {final_char_score:.2f}')
    print(f'final_pin_score : {final_pin_score:.2f}')