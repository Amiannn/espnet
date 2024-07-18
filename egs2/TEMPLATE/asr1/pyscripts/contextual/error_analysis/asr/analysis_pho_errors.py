import os
import torch
import random
import torchaudio
import numpy as np

from tqdm import tqdm
from fuzzywuzzy import fuzz

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

from text2phonemesequence import Text2PhonemeSequence

pattern_path = './exp/analysis/error_patterns_pho.json'
dump_path    = './exp/analysis'

if __name__ == '__main__': 
    datas    = read_json(pattern_path)
    patterns = datas['errors']

    scores_phos_list = []
    scores_list      = []
    for word in tqdm(list(patterns.keys())):
        word_pho = patterns[word]['pho']
        # print(f'word: {word}, pho: {word_pho}')
        errors = patterns[word]['errors']
        score_pho_list = []
        score_list     = []
        for hyp, hyp_pho in errors:
            score_pho = fuzz.ratio(word_pho, hyp_pho)
            score     = fuzz.ratio(word, hyp)
            score_pho_list.append(score_pho)
            score_list.append(score)
            # print(f'hyp: {hyp}, hyp_pho: {hyp_pho}, score_pho: {score_pho}, score_word: {score}')
        scores_phos_list.append(
            sum(score_pho_list) / len(score_pho_list)
        )
        scores_list.append(
            sum(score_list) / len(score_list)
        )
        print(f'_' * 30)
    scores_phos = sum(scores_phos_list) / len(scores_phos_list)
    scores      = sum(scores_list) / len(scores_list)

    print(f'scores_phos : {scores_phos}')
    print(f'scores_words: {scores}')