import torch
import argparse

# Utility functions for file I/O
from pyscripts.utils.fileio import read_yml
from espnet2.tasks.asr import ASRTask
from torch.nn.utils.rnn import pad_sequence

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
    conf = read_yml(config_path)
    conf['token_list'] = token_path
    conf['context_token_list'] = context_token_path if context_token_path is not None else token_path
    conf['input_size'] = conf.get('input_size', None)
    conf['specaug'] = conf.get('specaug', None)
    conf['normalize'] = conf.get('normalize', 'global_mvn')
    conf['frontend'] = frontend
    conf['frontend_conf'] = conf.get('frontend_conf', {'fs': '16k'})
    conf['init'] = None
    conf['normalize_conf'] = {'stats_file': stats_path} if stats_path is not None else {}
    conf['token_type'] = token_type if token_type is not None else conf.get('token_type', 'bpe')
    conf['context_token_type'] = context_token_type if context_token_type is not None else conf['token_type']
    conf['bpemodel'] = spm_path
    conf['context_bpemodel'] = context_spm_path
    conf['g2p'] = conf.get('g2p', None)
    conf['cleaner'] = conf.get('cleaner', None)
    conf['context_cleaner'] = conf.get('context_cleaner', None)
    conf['use_preprocessor'] = conf.get('use_preprocessor', True)
    conf['collate_fn_type'] = conf.get('collate_fn_type', 'contextual')

    conf['preprocessor'] = conf.get('preprocessor', 'contextual')
    conf['preprocessor_conf'] = preprocessor_conf
    conf['non_linguistic_symbols'] = conf.get('non_linguistic_symbols', None)

    conf['contextual_conf'] = conf.get('contextual_conf', {})
    conf['contextualizer_conf'] = conf.get('contextualizer_conf', {})

    conf['contextual_conf'].update(contextual_conf)
    conf['contextualizer_conf'].update({'use_local_attn_conv': use_local_attn_conv})

    args = argparse.Namespace(**conf)
    return args    

def map_tokens_to_words(token_ids, tokens, text, tokenizer):
    last_text = ""
    mapping = []
    collapse2token_idx = {}
    index = 0

    for idx in range(len(tokens)):
        # Get the token string up to the current token
        token_str = tokenizer.tokens2text(tokens[:idx + 1])
        if text.startswith(token_str):
            token_text = token_str[len(last_text):]
            index = len(last_text)
            last_text = token_str
        mapping.append([token_ids[idx], token_text, index, idx])

    for i in range(len(mapping)):
        token_idx = mapping[i][-1]
        token_start = mapping[i][-2]
        token_text = mapping[i][1]
        for char_pos in range(token_start, token_start + len(token_text)):
            collapse2token_idx[char_pos] = token_idx

    return collapse2token_idx

def find_phrase_positions(sentence, phrase):
    positions = []
    index = sentence.find(phrase)
    while index != -1:
        positions.append(index)
        index = sentence.find(phrase, index + 1)
    return positions

def find_contexts_with_positions(sentence, entity_phrases):
    detected_phrases = []
    for i, phrase in enumerate(entity_phrases):
        # Remove spaces in phrase to match the text correctly
        positions = find_phrase_positions(sentence, phrase)
        if positions:
            detected_phrases.append((i, positions))
    return detected_phrases

def build_label_for_cross_entropy_loss(
        token_ids_batch, 
        ent_datas, 
        tokenizer, 
        idconverter, 
        pad_value=0
    ):
    batch_size, seq_length = token_ids_batch.shape
    labels = torch.full((batch_size, seq_length), 0, dtype=torch.long)
    labels[token_ids_batch == pad_value] = pad_value
    for i in range(batch_size):
        token_ids       = token_ids_batch[i]
        valid_token_ids = token_ids[token_ids != pad_value].tolist()
        tokens          = idconverter.ids2tokens(valid_token_ids, skip_special_tokens=False)
        text            = tokenizer.tokens2text(tokens)

        char2token_idx = map_tokens_to_words(valid_token_ids, tokens, text, tokenizer)
        for ent_idx, positions in find_contexts_with_positions(text, ent_datas):
            for pos in positions:
                index = char2token_idx[pos]
                labels[i, index] = ent_idx + 1
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
    tokenizer = preprocessor.tokenizer
    idconverter = preprocessor.token_id_converter
    
    # Define your batch of texts
    texts = [
        "這 個 部分 也 是 擴大我們 在 顧客 的 方面 擴大 了 尤其 corporate 這 個 部分 擴大我們",
        "另一個文本示例，其中包括擴大我們和其他內容",
        # Add more texts as needed
    ]

    ent_datas = ["擴大我們", "corporate", "其他內容"]

    token_ids_list = []
    for text in texts:
        tokens = tokenizer.text2tokens(text)
        token_ids = idconverter.tokens2ids(tokens)
        token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))

    # Pad the sequences to the same length
    pad_value = -1  # Adjust this if your tokenizer uses a different pad token ID
    token_ids_padded = pad_sequence(token_ids_list, batch_first=True, padding_value=pad_value)

    # Generate labels for the batch
    labels = build_label_for_cross_entropy_loss(token_ids_padded, ent_datas, tokenizer, idconverter, pad_value=pad_value)
    print(f'labels: {labels}')
