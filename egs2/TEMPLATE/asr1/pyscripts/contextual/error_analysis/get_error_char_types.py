import os
import json

from jiwer import cer
from tqdm  import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.contextual.utils.aligner import CheatDetector
from pyscripts.contextual.utils.aligner import align_to_index

ref_path = '/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/data/zh_test/text'
hyp_path = '/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe4500_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/zh_test/text'

if __name__ == '__main__':
    # hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    # refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    hyps = [[d[0], list("".join([i for i in d[1:] if i != '']))] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], list("".join(d[1:]))] for d in read_file(ref_path, sp=' ')]

    outputs = {}
    for ref, hyp in zip(refs, hyps):
        uid = ref[0]
        ref = ref[1]
        hyp = hyp[1]
        chunks = align_to_index(ref, hyp)
        output = []
        for chunk in chunks:
            _, _, rindex, hindexis = chunk
            wref = "".join([ref[i] for i in rindex])
            href = "".join([hyp[i] for i in hindexis])
            if wref != href and href != '':
                output.append([wref, href])
        if len(output) > 0:
            outputs[uid] = output

    output_path = 'aishell_insertion.json'
    write_json(output_path, outputs)