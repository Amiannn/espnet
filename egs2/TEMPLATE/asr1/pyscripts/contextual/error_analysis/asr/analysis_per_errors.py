import os
import torch
import random
import torchaudio
import numpy as np

import jiwer

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

pattern_path = './exp/analysis/pho_result.json'
dump_path    = './exp/analysis'

if __name__ == '__main__': 
    datas     = read_json(pattern_path)
    pho_datas = datas['pho']

    ref_phos = [d[0] for d in pho_datas]
    hyp_phos = [d[1] for d in pho_datas]

    out = jiwer.process_words(
        ref_phos,
        hyp_phos,
    )

    print(jiwer.visualize_alignment(out))