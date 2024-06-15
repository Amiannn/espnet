import os
import json
import jieba

from jiwer import cer
from jiwer import wer
from jiwer import mer
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
ref_path       = './data/test/text'
# hyp_path       = "../asr1/exp/asr_train_asr_transducer_conformer_raw_bpe5000_use_wandbtrue_sp_suffix/decode_asr_rnnt_transducer_bs5_asr_model_valid.loss.ave_10best/test/text"
# hyp_path       = "/share/nas165/amian/experiments/speech/espnet/egs2/esun/asr1/exp/asr_train_asr_transducer_conformer_raw_bpe5000_use_wandbtrue_sp_suffix/decode_asr_rnnt_transducer_greedy_asr_model_valid.loss.ave_10best/test/text"
# hyp_path       = "./exp/asr_transducer/contextual_xphone_adapter_suffix/decode_contextual_xphone_adapter_greedy_asr_model_valid.loss.ave_10best/test/text"
hyp_path       = "/share/nas165/amian/experiments/speech/espnet/egs2/esun/asr1_contextual/exp/asr_transducer/contextual_adapter_annhnw_suffix/decode_contextual_adapter_bs10_asr_model_valid.loss.ave_10best/test/text"

def check_passed(indexis, memory):
    for index in indexis:
        if index in memory:
            return True
    return False

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def preprocess_sent(words):
    new_words = []
    for word in words:
        if isEnglish(word):
            new_words.append(word)
        else:
            new_words.extend(list(word))
    return new_words

def preprocess_sent_sen(words):
    new_words = []
    new_word  = ""
    for word in words:
        if isEnglish(word):
            if new_word != "":
                new_words.append(new_word)
                new_word = ""
            new_words.append(word)
        else:
            new_word += word
    if new_word != "":
        new_words.append(new_word)
    return ' '.join(new_words)

def find_rareword(sent, rarewords):
    blist = []
    for word in rarewords:
        if isEnglish(word):
            if  (f' {word} ' in sent) or \
                (f'{word} ' == sent[:len(word) + 1]) or \
                (f' {word}' == sent[-(len(word) + 1):]) or \
                (word == sent):
                blist.append(word)
        else:
            if word in sent.replace(' ', ''):
                blist.append(word)
    return blist

if __name__ == '__main__':
    # uttblist = read_json(utt_blist_path)
    rareword = [d[0] for d in read_file(rareword_list, sp=' ')]
    # print(rareword[:10])

    hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    ref_rareword_sents = []
    hyp_rareword_sents = []
    ref_common_sents   = []
    hyp_common_sents   = []
    ref_rareword_engs  = []
    hyp_rareword_engs  = []
    ref_rareword_zhs   = []
    hyp_rareword_zhs   = []
    ref_sents          = []
    hyp_sents          = []
    error_pattern      = {word: {} for word in rareword}
    entity_freq        = {word: 0 for word in rareword}

    for ref, hyp in zip(refs, hyps):
        # blist = uttblist[ref[0]]
        blist     = find_rareword(" ".join(ref[1]), rareword)
        ref_words = ref[1]
        hyp_words = hyp[1]
        print(f'uid : {ref[0]}')
        print(f'ref: {ref_words}')
        print(f'hyp: {hyp_words}')
        if len(hyp_words) == 0:
            print(f'error: {ref[0]} has zero lengths!')
            continue
        
        chunks = align_to_index(ref_words, hyp_words)
        # print(f'chunks: {chunks}')
        ref_words = preprocess_sent(ref[1])
        hyp_words = preprocess_sent(hyp[1])
        ref_sents.append(' '.join(ref_words))
        hyp_sents.append(' '.join(hyp_words))
        ref_rareword_sent = []
        hyp_rareword_sent = []
        ref_rareword_eng  = []
        hyp_rareword_eng  = []
        ref_rareword_zh   = []
        hyp_rareword_zh   = []
        ref_common_sent   = []
        hyp_common_sent   = []
        passed_index      = []
        # print(f'blist: {blist}')
        # print(chunks)
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref = wref.replace('-', '')
            _whyps = preprocess_sent_sen([whyp.replace('-', '') for whyp in whyps])
            # print(f'wref: {wref}, whyps: {whyps}, _whyps: {_whyps}')
            whyps = _whyps
            if wref in blist:
                entity_freq[wref] += 1
                if wref != whyps:
                    if whyps not in error_pattern[wref]:
                        error_pattern[wref][whyps] = 1
                    else:
                        error_pattern[wref][whyps] += 1
                ref_rareword_sent.append(wref)
                hyp_rareword_sent.append(whyps)
                if isEnglish(wref):
                    ref_rareword_eng.append(wref)
                    hyp_rareword_eng.append(whyps)
                else:
                    ref_rareword_zh.append(wref)
                    hyp_rareword_zh.append(whyps)

            elif not check_passed(hindexis, passed_index):
                ref_common_sent.append(wref)
                hyp_common_sent.append(whyps)
                passed_index.extend(hindexis)
        # print("_" * 30)
        if len(ref_rareword_sent) > 0:
            ref_rareword_sents.append(' '.join(ref_rareword_sent))
            hyp_rareword_sents.append(' '.join(hyp_rareword_sent))
        else:
            ref_rareword_sents.append('correct')
            hyp_rareword_sents.append('correct')
        if len(ref_common_sent) > 0:
            ref_common_sents.append(' '.join(ref_common_sent))
            hyp_common_sents.append(' '.join(hyp_common_sent))
        else:
            ref_common_sents.append('correct')
            hyp_common_sents.append('correct')
        
        if len(ref_rareword_eng) > 0:
            ref_rareword_engs.append(' '.join(ref_rareword_eng))
            hyp_rareword_engs.append(' '.join(hyp_rareword_eng))
        if len(ref_rareword_zh) > 0:
            ref_rareword_zhs.append(' '.join(ref_rareword_zh))
            hyp_rareword_zhs.append(' '.join(hyp_rareword_zh))
    
    
    ref_rareword_sents = [' '.join(preprocess_sent(ref_rare.split(' '))) for ref_rare in ref_rareword_sents]
    hyp_rareword_sents = [' '.join(preprocess_sent(hyp_rare.split(' '))) for hyp_rare in hyp_rareword_sents]

    ref_rareword_sents = [" ".join(ref_rare.split()) for ref_rare in ref_rareword_sents]
    hyp_rareword_sents = [" ".join(hyp_rare.split()) for hyp_rare in hyp_rareword_sents]

    ref_common_sents = [' '.join(preprocess_sent(ref_common.split(' '))) for ref_common in ref_common_sents]
    hyp_common_sents = [' '.join(preprocess_sent(hyp_common.split(' '))) for hyp_common in hyp_common_sents]

    ref_common_sents = [" ".join(ref_common.split()) for ref_common in ref_common_sents]
    hyp_common_sents = [" ".join(hyp_common.split()) for hyp_common in hyp_common_sents]

    all_mer      = mer(ref_sents, hyp_sents)
    rareword_mer = mer(ref_rareword_sents, hyp_rareword_sents)
    common_mer   = mer(ref_common_sents, hyp_common_sents)

    eng_rareword_mer = wer(ref_rareword_engs, hyp_rareword_engs)
    zh_rareword_mer  = cer(ref_rareword_zhs, hyp_rareword_zhs)

    # print(ref_rareword_sents[:10])
    # print(hyp_rareword_sents[:10])

    print(f'all_mer         : {all_mer*100:.2f}%')
    print(f'rareword_mer    : {rareword_mer*100:.2f}%')
    print(f'common_mer      : {common_mer*100:.2f}%')
    print(f'rareword_eng_wer: {eng_rareword_mer*100:.2f}%')
    print(f'rareword_zh_cer : {zh_rareword_mer*100:.2f}%')

    output_path = './exp/test/ref_sents'
    write_file(output_path, [[d] for d in ref_sents], sp='')
    output_path = './exp/test/hyp_sents'
    write_file(output_path, [[d] for d in hyp_sents], sp='')

    output_path = './exp/test/ref_common_sents'
    write_file(output_path, [[d] for d in ref_common_sents], sp='')
    output_path = './exp/test/hyp_common_sents'
    write_file(output_path, [[d] for d in hyp_common_sents], sp='')

    output_path = './exp/test/ref_rareword_sents'
    write_file(output_path, [[d] for d in ref_rareword_sents], sp='')
    output_path = './exp/test/hyp_rareword_sents'
    write_file(output_path, [[d] for d in hyp_rareword_sents], sp='')

    output_path = './exp/test/ref_rareword_engs'
    write_file(output_path, [[d] for d in ref_rareword_engs], sp='')
    output_path = './exp/test/hyp_rareword_engs'
    write_file(output_path, [[d] for d in hyp_rareword_engs], sp='')

    output_path = './exp/test/ref_rareword_zhs'
    write_file(output_path, [[d] for d in ref_rareword_zhs], sp='')
    output_path = './exp/test/hyp_rareword_zhs'
    write_file(output_path, [[d] for d in hyp_rareword_zhs], sp='')

    output_path = './exp/test/error_pattren_asr.tsv'
    # error_pattern = [[key, str(entity_freq[key]), ", ".join([d for d in error_pattern[key] if d != ''])] for key in error_pattern]
    
    error_pattern_list = []
    for key in error_pattern:
        pattern_list = []
        error_sum = 0
        for skey in error_pattern[key]:
            _skey = '_' if skey == '' else skey
            error_sum += error_pattern[key][skey]
            # pattern_list.append(f'{_skey} ({error_pattern[key][skey]})')
            pattern_list.append([error_pattern[key][skey], _skey])
        pattern_list = sorted(pattern_list, reverse=True)
        pattern_list = [f'{d[1]} ({d[0]})' for d in pattern_list]
        
        freq = entity_freq[key]
        error_rate = error_sum / freq if freq > 0 else 0.0
        error_pattern_list.append([key, freq, f'{(error_rate):.2}', error_sum, ", ".join(pattern_list)])
    error_pattern_list = sorted(error_pattern_list, key=lambda d: d[1], reverse=True)
    error_pattern_list = [[str(_d) for _d in d] for d in error_pattern_list]
    write_file(output_path, sorted(error_pattern_list), sp='\t')
