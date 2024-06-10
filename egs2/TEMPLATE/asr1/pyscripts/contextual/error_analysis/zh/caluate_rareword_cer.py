import os
import json
import jieba

from jiwer import cer
from tqdm  import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

rareword_list  = './local/contextual/rarewords/rareword_f10_test.txt'
utt_blist_path = './dump/raw/test/uttblist_idx'
ref_path       = './dump/raw/test/text'
hyp_path       = '/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__annhnw__mediumbatch/decode_asr_conformer_adapter_bs10_asr_model_valid.acc.ave_10best/test/text'

def check_passed(indexis, memory):
    for index in indexis:
        if index in memory:
            return True
    return False

if __name__ == '__main__':
    rareword = [d[0] for d in read_file(rareword_list, sp=' ')]
    uttblist = {d[0]: [rareword[int(i)] for i in d[1:] if i != ''] for d in read_file(utt_blist_path, sp=' ')}

    # hyps = [[d[0], [i for i in d[1:] if i != ''][0]] for d in read_file(hyp_path, sp=' ')]
    hyps = {d[0]: d[1:] for d in read_file(hyp_path, sp=' ')}
    # hyp2s = {d[0]: d[1:] for d in read_file(hyp2_path, sp=' ')}
    # refs = [[d[0], d[1:][0]] for d in read_file(ref_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    ref_rareword_sents = []
    hyp_rareword_sents = []
    ref_common_sents   = []
    hyp_common_sents   = []
    ref_sents          = []
    hyp_sents          = []

    count = 0
    for ref in refs:
        uid = ref[0]
        ref = ref[1]
        if uid not in hyps:
            print(f'{uid}')
            continue
        hyp = hyps[uid]
        blist  = uttblist[uid]
        # print(f'ref: {ref}')
        # print(f'hyp: {hyp}')
        ref = list(jieba.cut(ref[0]))
        hyp = list(jieba.cut(hyp[0]))
        
        # print(f'blist: {blist}')
        chunks = align_to_index(ref, hyp)
        # print(f'chunks: {chunks}')
        ref_sents.append(''.join(ref))
        hyp_sents.append(''.join(hyp))
        ref_rareword_sent = []
        hyp_rareword_sent = []
        ref_common_sent   = []
        hyp_common_sent   = []
        passed_index      = []
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref = wref.replace('-', '')
            whyps = ''.join(whyps).replace('-', '')
            if wref in blist:
                ref_rareword_sent.append(wref)
                hyp_rareword_sent.append(whyps)
            elif not check_passed(hindexis, passed_index):
                ref_common_sent.append(wref)
                hyp_common_sent.append(whyps)
                passed_index.extend(hindexis)
        if len(ref_rareword_sent) > 0:
            ref_rareword_sents.append(''.join(ref_rareword_sent))
            hyp_rareword_sents.append(''.join(hyp_rareword_sent))
        if len(ref_common_sent) > 0:
            ref_common_sents.append(''.join(ref_common_sent))
            hyp_common_sents.append(''.join(hyp_common_sent))
        count += 1
    all_cer      = cer(ref_sents, hyp_sents)
    rareword_cer = cer(ref_rareword_sents, hyp_rareword_sents)
    common_cer   = cer(ref_common_sents, hyp_common_sents)

    print(f'all_cer     : {all_cer}')
    print(f'rareword_cer: {rareword_cer}')
    print(f'common_cer  : {common_cer}')

    # output_path = './ref_common_sents'
    # write_file(output_path, [[d] for d in ref_common_sents], sp='')
    # output_path = './hyp_common_sents'
    # write_file(output_path, [[d] for d in hyp_common_sents], sp='')

