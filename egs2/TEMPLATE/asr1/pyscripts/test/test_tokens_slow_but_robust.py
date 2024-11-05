import jieba
import torch

import argparse
import sentencepiece as spm

# Utility functions for file I/O
from pyscripts.utils.fileio import read_yml
from pyscripts.utils.fileio import read_file, write_file

from espnet2.tasks.asr import ASRTask

def load_config(
    config_path,
    contextual_conf,
    token_path, 
    context_token_path, 
    frontend=None, 
    stats_path=None, 
    spm_path=None, 
    context_spm_path=None, 
    use_local_attn_conv=False,
    token_type=None,
    context_token_type=None,
    preprocessor_conf={},
):
    conf                       = read_yml(config_path)
    conf['token_list']         = token_path
    conf['context_token_list'] = context_token_path if context_token_path is not None else token_path
    conf['input_size']         = None if 'input_size' not in conf else conf['input_size']
    conf['specaug']            = None if 'specaug' not in conf else conf['specaug']
    conf['normalize']          = 'global_mvn' if 'normalize' not in conf else conf['normalize']
    conf['frontend']           = frontend
    conf['frontend_conf']      = {'fs': '16k'} if 'frontend_conf' not in conf else conf['frontend_conf']
    # conf['ctc_conf']           = get_default_kwargs(CTC)
    conf['init']               = None
    conf['normalize_conf']     = {'stats_file': stats_path} if stats_path is not None else {}
    conf['token_type']         = 'bpe' if 'token_type' not in conf else conf['token_type']
    conf['token_type']         = token_type if token_type is not None else conf['token_type']
    conf['context_token_type'] = context_token_type if context_token_type is not None else conf['token_type']
    conf['bpemodel']           = spm_path
    conf['context_bpemodel']   = context_spm_path
    conf['g2p']                = None if 'g2p' not in conf else conf['g2p']
    conf['cleaner']            = None if 'cleaner' not in conf else conf['cleaner']
    conf['context_cleaner']    = None if 'context_cleaner' not in conf else conf['context_cleaner']
    conf['use_preprocessor']   = True if 'use_preprocessor' not in conf else conf['use_preprocessor']
    conf['collate_fn_type']    = 'contextual' if 'collate_fn_type' not in conf else conf['collate_fn_type']

    conf['preprocessor']           = 'contextual' if 'preprocessor' not in conf else conf['preprocessor']
    conf['preprocessor_conf']      = preprocessor_conf
    conf['non_linguistic_symbols'] = None if 'non_linguistic_symbols' not in conf else conf['non_linguistic_symbols']

    if 'contextual_conf' not in conf:
        conf['contextual_conf'] = {}
    if 'contextualizer_conf' not in conf:
        conf['contextualizer_conf'] = {}
            
    conf['contextual_conf'].update(contextual_conf)
    conf['contextualizer_conf'].update({'use_local_attn_conv': use_local_attn_conv})

    args = argparse.Namespace(**conf)
    return args    

def map_tokens_to_words(token_ids, tokens, text, tokenizer, idconverter):
    last_text  = ""
    token_text = ""
    mapping    = []

    collapse2token_idx = {}

    index = 0
    for idx in range(len(tokens)):
        # Get the token string
        token_str = tokenizer.tokens2text(tokens[:idx + 1])
        if text.find(token_str) == 0:
            token_text = token_str[len(last_text):]
            index      = len(last_text)
            last_text  = token_str
        mapping.append([token_ids[idx], token_text, index, idx])
    
    for i in range(len(mapping) - 1):
        now_idx  = mapping[i][-2]
        next_idx = mapping[i + 1][-2]

        for j in range(now_idx, next_idx):
            collapse2token_idx[j] = mapping[i][-1]

    return collapse2token_idx

def find_phrase_positions(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    positions = []
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            positions.append(i)
    return positions

def align_sequences(sent_a, sent_b):
    collapse_a     = [idx for idx, char in enumerate(sent_a) if char != " "]
    idx2collapse_a = {idx: i for i, idx in enumerate(collapse_a)}
    collapse2idx_a = {i:idx for i, idx in enumerate(collapse_a)}
    
    collapse_b     = {idx: char for idx, char in enumerate(sent_b) if char != " "}
    idx2collapse_b = {idx: i for i, idx in enumerate(collapse_b)}
    collapse2idx_b = {i:idx for i, idx in enumerate(collapse_b)}

    return idx2collapse_a, collapse2idx_a, idx2collapse_b, collapse2idx_b

def find_rare_words_with_positions(sentence, entity_phrases):
    # Segment the sentence using jieba
    sentence_no_space  = sentence.replace(" ", "")
    segmented_sentence = list(jieba.cut(sentence_no_space))

    sent_i2c, sent_c2i, seg_sent_i2c, seg_sent_c2i = align_sequences(sentence, " ".join(segmented_sentence))

    # Build a list of cumulative character indices
    char_indices = []
    index = 0
    for word in segmented_sentence:
        char_indices.append(index)
        index += (len(word) + 1)

    # List to keep track of detected phrases and their positions
    detected_phrases = []

    # Check if each phrase is present in the segmented sentence
    for i, phrase in enumerate(entity_phrases):
        # Remove spaces in phrase and segment it
        phrase_no_space = "".join(phrase.split(' '))
        segmented_phrase = list(jieba.cut(phrase_no_space))
        phrase_len = len(segmented_phrase)

        positions_in_segmented_sentence = find_phrase_positions(segmented_phrase, segmented_sentence)

        if positions_in_segmented_sentence:
            # For each position, get the starting character index in the original sentence
            positions_in_segmented_sentence = [char_indices[idx] for idx in positions_in_segmented_sentence]
            positions_in_sentence = [sent_c2i[seg_sent_i2c[idx]] for idx in positions_in_segmented_sentence]
            detected_phrases.append((i, positions_in_sentence))
    return detected_phrases

def build_label_for_cross_entropy_loss(token_ids, ent_datas):
    token_ids = token_ids.tolist()
    tokens    = idconverter.ids2tokens(token_ids, skip_special_tokens=False)
    text      = tokenizer.tokens2text(tokens)

    char2token_idx = map_tokens_to_words(token_ids, tokens, text, tokenizer, idconverter)

    # (-1) for padding value
    labels = torch.zeros(len(token_ids), dtype=torch.long) - 1
    data   = find_rare_words_with_positions(text, ent_datas)
    print(data)
    
    for ent_idx, positions in data:
        for pos in positions:
            index = char2token_idx[pos]
            print(f'index: {index}')
            labels[index] = ent_idx + 1
    return labels

if __name__ == '__main__':
    config_path = "./conf/contextual/whisper/train_asr_whisper_medium_contextual_adapter_decoder.yaml"
    spm_path = "whisper_multilingual"
    context_spm_path = spm_path
    token_path = "./data/zh_token_list/whisper_multilingual/tokens.txt"
    context_token_path = token_path
    stats_path = None
    
    args = load_config(
        config_path=config_path,
        contextual_conf={},
        token_path=token_path, 
        context_token_path=context_token_path, 
        frontend='default', 
        stats_path=stats_path, 
        spm_path=spm_path, 
        context_spm_path=context_spm_path, 
        token_type='whisper_multilingual',
        context_token_type='whisper_multilingual',
    )

    preprocessor = ASRTask.build_preprocess_fn(args, train=False)

    tokenizer   = preprocessor.tokenizer
    idconverter = preprocessor.token_id_converter
    
    text      = "這 個 部分 也 是 擴大 我們 在 顧客 的 方面 擴大 了 尤其 corporate 這 個 部分 擴大我們"
    ent_datas = ["擴大我們", "corporate"]

    
    tokens    = tokenizer.text2tokens(text)
    token_ids = idconverter.tokens2ids(tokens)
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # for phrase, token, idx, i in mapping:
    #     print(f'{phrase}\t{token}, {idx}, {i}')

    labels = build_label_for_cross_entropy_loss(token_ids, ent_datas)
    print(f'labels: {labels}')