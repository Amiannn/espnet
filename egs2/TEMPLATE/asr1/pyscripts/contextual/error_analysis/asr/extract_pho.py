import os
import torch
import random
import torchaudio
import numpy as np

from tqdm import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

from text2phonemesequence import Text2PhonemeSequence

pattern_path = './exp/analysis/error_patterns.json'
dump_path    = './exp/analysis'
pho_dict     = {}

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='zho-s', is_cuda=False)
# text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=False)

def get_pho(word):
    if word in pho_dict:
        return pho_dict[word]
    word_pho = text2phone_model.infer_sentence(word.lower())
    pho_dict[word] = word_pho
    return word_pho

if __name__ == '__main__': 
    datas    = read_json(pattern_path)
    patterns = datas['errors']
    patterns_pho = {}
    for word in tqdm(list(patterns.keys())):
        word_pho = get_pho(word)
        patterns_pho[word] = {
            'pho'   : word_pho,
            'errors': []
        }
        for hyp in patterns[word]:
            hyp_pho = get_pho(hyp)
            patterns_pho[word]['errors'].append(
                [hyp, hyp_pho]
            )
    output_path = os.path.join(dump_path, 'error_patterns_pho.json')
    write_json(output_path, {
        'metadata': datas['metadata'],
        'errors': patterns_pho
    })