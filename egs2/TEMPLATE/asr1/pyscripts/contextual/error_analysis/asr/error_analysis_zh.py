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

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

hyp_path  = "./exp/asr_train_asr_transducer_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_rnnt_transducer_greedy_asr_model_valid.loss.ave_10best/test/text"
ref_path  = "./dump/raw/zh_test/text"
dump_path = './exp/analysis'

if __name__ == '__main__':
    hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    patterns = {}
    for ref, hyp in zip(refs, hyps):
        ref_words = ref[1]
        hyp_words = hyp[1][0]
        print(hyp_words)
        hyp_words = list(jieba.cut(hyp_words))
        if len(hyp_words) == 0:
            print(f'error: {ref[0]} has zero lengths!')
            continue
        chunks = align_to_index(ref_words, hyp_words)
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            whyps = " ".join(whyps)
            wref  = wref.replace('-', '')
            whyps = whyps.replace('-', '')
            if wref != whyps:
                if wref in patterns:
                    patterns[wref].append(whyps)
                else:
                    patterns[wref] = [whyps]
    
    output_path = os.path.join(dump_path, 'error_patterns.json')
    write_json(output_path, {
        'metadata': {
            'ref_path': ref_path,
            'hyp_path': hyp_path,
        },
        'errors': patterns
    })