import os
import torch
import jieba
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

from text2phonemesequence import Text2PhonemeSequence

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='zho-s', is_cuda=False)

hyp_path  = "./exp/asr_train_asr_transducer_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_rnnt_transducer_greedy_asr_model_valid.loss.ave_10best/test/text"
ref_path  = "./dump/raw/zh_train_sp/text"
dump_path = './exp/analysis'

cache = {}
def get_phone(words):
    phos = []
    for word in words:
        if word in cache:
            phos.append(cache[word])
        else:
            pho = text2phone_model.infer_sentence(word.lower())
            cache[word] = pho
            phos.append(pho)
    return " ▁ ".join(phos)

if __name__ == '__main__':
    # hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ') if not d[0].startswith("sp")]
    
    datas = []
    # for ref, hyp in tqdm(zip(refs, hyps)):
    for ref in tqdm(refs):
        ref_words = " ".join(ref[1])
        # hyp_words = " ".join(hyp[1])
        # hyp_words = " ".join(list(jieba.cut(hyp_words)))
        print()
        print(f'ref: {ref_words}')
        # print(f'hyp: {hyp_words}')
        ref_pho = get_phone(ref_words.split(' '))
        # hyp_pho = get_phone(hyp_words.split(' '))
        print(f'ref_pho: {ref_pho}')
        # print(f'hyp_pho: {hyp_pho}')

        # datas.append([ref_pho, hyp_pho])
        datas.append(ref_pho)

    output_path = os.path.join(dump_path, 'pho_result_train.json')
    write_json(output_path, {
        'metadata': {
            'ref_path': ref_path,
            # 'hyp_path': hyp_path,
        },
        'pho': datas
    })