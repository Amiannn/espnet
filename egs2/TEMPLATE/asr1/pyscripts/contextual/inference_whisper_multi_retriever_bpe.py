import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import groupby

# Utility functions for file I/O
from pyscripts.utils.fileio import read_file, read_json, read_pickle, write_file, write_json, write_pickle

# ESPnet-related imports
from pyscripts.contextual.utils.model import load_espnet_model
from pyscripts.contextual.utils.rnnt_decode import infernece
from pyscripts.contextual.utils.rnnt_alignment import forward_backward as force_alignment
from pyscripts.contextual.utils.visualize import plot_attention_map, plot_tsne, plot_gate

from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

from espnet2.asr.contextualizer.func.contextual_retriever_func import topk_decode, retrieve_ctc_decode

# Seed setting for reproducibility
seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# ---- Utility Functions ---- #

def median_filter_tensor(input_tensor, kernel_size):
    """Applies a median filter over a tensor"""
    B, T, C = input_tensor.shape
    pad = kernel_size // 2
    input_tensor_padded = F.pad(input_tensor, (0, 0, pad, pad), mode='reflect')
    unfolded = input_tensor_padded.unfold(1, kernel_size, 1)
    return torch.median(unfolded, dim=2).values

def get_token_list(token_id_converter):
    """Retrieve token list from the token ID converter"""
    vocab_size = token_id_converter.get_num_vocabulary_size()
    return [token_id_converter.ids2tokens([i])[0] if len(token_id_converter.ids2tokens([i])) > 0 else '' for i in range(vocab_size)]

def retriever_decode(ys_hat, char_list, blank_index=0):
    """Decode the output sequences"""
    sequence_prediction = [int(x[0]) for y in ys_hat for x in groupby(y) if int(x[0]) != -1 and int(x[0]) != blank_index]
    return ", ".join([char_list[int(idx)] for idx in sequence_prediction])

def visualize(logp, attention, ctc_prediction, text, target, biasing_list, speech, blank_id, token_list, debug_path, utterance_id):
    """Visualize the attention maps and predictions"""
    frame2align = {i: token_list[p] if p != 0 else ' ' for i, p in enumerate(ctc_prediction)} if ctc_prediction is not None else {}
    plot_attention_map(frame2align, attention, text, biasing_list, debug_path, utterance_id)

@torch.no_grad()
def forward(model, speech, speech_length, context_data, tokens, text, token_list):
    """Forward pass through the model and contextualization"""
    encoder_output, encoder_output_lengths = model.encode(speech, speech_length)
    
    encoder_projection = None
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_RETRIEVER:
        context_probabilities, encoder_projection = model.contextualizer(
            query=encoder_output,
            query_ilens=encoder_output_lengths,
            context_subword=context_data['blist'],
            context_subword_ilens=context_data['ilens'],
            context_phone=context_data['blist_xphone'],
            context_phone_ilens=context_data['blist_xphone_ilens'],
            return_model_proj=True
        )
        
        # context_prob_sw = torch.softmax(model.contextualizer.retriever.sw_score, dim=-1)
        # context_prob_pho = torch.softmax(model.contextualizer.retriever.ph_score, dim=-1)

        prediction = topk_decode(context_probabilities, biasing_list, idx_blank=0, top_k=5, threshold=0.9)
    
    ctc_prediction = None
    if model.ctc is not None:
        x = encoder_projection if encoder_projection is not None else encoder_output
        ctc_prediction = model.ctc.argmax(x).squeeze(0)
    
    predicted_hypothesis = retrieve_ctc_decode(model.ctc.ctc_lo(x), token_list, idx_blank=0, threshold=0.0)
    predicted_hypothesis = "".join([d[1] for d in predicted_hypothesis]).replace("▁", ' ')
    
    return None, None, context_probabilities, ctc_prediction, {
        'text': text,
        'hyp': predicted_hypothesis,
        'result': prediction,
    }

# ---- Main Functionality ---- #

if __name__ == "__main__":
    # File paths
    spm_path = "./data/token_list/bpe_unigram5000suffix/bpe.model"
    token_path = "./data/token_list/bpe_unigram5000suffix/tokens.txt"
    model_conf = "./conf/contextual/whisper/train_asr_whisper_medium_xdotproduct_contextual_retriever.yaml"
    model_path = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_suffix/74epoch.pth"
    stats_path = "./exp/asr_stats_raw_bpe5000_sp_suffix/train/feats_lengths_stats.npz"
    rareword_path = "./local/contextual/rarewords/esun.entity.txt"
    speech_scp_path = "./dump/raw/test/wav.scp"
    biasing_list_path = "./dump/raw/test/uttblist_idx_entity"
    biasing_list_xphone_path = "./local/contextual/ssl_features/esun.entity.xphone.seq.pt"
    reference_path = "./data/test/text"
    
    # Debug directory setup
    folder_name = model_path.split('/')[-1].split('.')[0]
    debug_path = os.path.join("/".join(model_path.split('/')[:-1]), 'debug', folder_name)
    if not os.path.isdir(debug_path):
        os.makedirs(debug_path)

    # Load reference texts
    reference_texts = {d[0]: " ".join(d[1:]) for d in read_file(reference_path, sp=' ')}

    # Model loading and configuration
    data_path_and_name_and_type = [(speech_scp_path, 'speech', 'kaldi_ark'), (biasing_list_path, 'uttblist_idx', 'multi_columns_text')]
    contextual_conf = {
        'contextual_type': 'context_sampler',
        'context_list_path': rareword_path,
        'context_phone_embedding_path': biasing_list_xphone_path,
        'max_batch_disrupt_context': 10,
        'sub_context_list_dropout': 0.0,
        'warmup_epoch': 0,
        'use_no_context_token': True,
        'context_prompt_has_context_template': '主題為:',
        'context_prompt_no_context_template': '開始吧',
    }

    model, loader, contextual_processor = load_espnet_model(
        model_conf, contextual_conf, token_path, 'default', stats_path, spm_path, model_path,
        data_path_and_name_and_type, use_local_attn_conv=False, return_contextual_processor=True,
    )

    # Prepare tokenizer and token list
    preprocessor = loader.dataset.preprocess
    tokenizer = preprocessor.tokenizer
    token_id_converter = preprocessor.token_id_converter
    token_list = get_token_list(token_id_converter) + ['<no-context>']

    # Model evaluation
    model.eval()
    results = {}
    count = 0
    for data in loader:
        if count >= 20:
            break
        count += 1

        uid = data[0][0]
        data = data[1]
        context_data = data['contexts']
        speech = data['speech']
        speech_length = data['speech_lengths']
        text = reference_texts[uid]
        biasing_list = context_data['blist']
        label_ctc = context_data['label_ctc']

        _biasing_list = [tokenizer.tokens2text([token_list[word] for word in rareword if word != -1]) for rareword in biasing_list]
        biasing_list = _biasing_list

        tokens = torch.tensor(preprocessor._text_process({'text': text})['text']).long().unsqueeze(0)

        logp, target, attention, ctc_prediction, prediction = forward(model, speech, speech_length, context_data, tokens, text, token_list)
        results[uid] = prediction
        
        # for attention, tag in zip(attentions, ['combine', 'sw', 'pho']):
        #     visualize(logp, attention, ctc_prediction, text, target, biasing_list, speech, model.blank_id, token_list, debug_path, f'{uid}_{tag}')
        visualize(logp, attention, ctc_prediction, text, target, biasing_list, speech, model.blank_id, token_list, debug_path, f'{uid}')

    # Save results
    output_path = os.path.join(debug_path, 'predict.json')
    write_json(output_path, results)