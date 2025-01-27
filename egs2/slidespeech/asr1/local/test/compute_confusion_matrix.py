import os
import random
import numpy as np
import torch
import math
import torch.nn.functional as F
from typing import List, Any
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

from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    decode_topk_tokens,
)

# ---- Utility Functions ---- #
def decode_ctc_predictions(
    ctc_probs: torch.Tensor,
    vocabulary: List[str],
    blank_index: int = 0,
    threshold: float = 0.0,
    **kwargs,
) -> List[List[List[Any]]]:
    """
    Decodes CTC output probabilities to retrieve token sequences.

    Args:
        ctc_probs (torch.Tensor): The CTC probability tensor of shape (batch_size, seq_length, num_classes).
        vocabulary (List[str]): List of token strings corresponding to class indices.
        blank_index (int, optional): Index of the blank token in CTC. Defaults to 0.
        threshold (float, optional): Probability threshold for token inclusion. Defaults to 0.0.

    Returns:
        List[List[List[Any]]]: Decoded token sequences for each batch.
    """
    predicted_indices = ctc_probs.argmax(dim=-1).cpu()
    predicted_probs = ctc_probs.max(dim=-1).values.cpu()
    batch_sequences = []

    for b in range(predicted_indices.shape[0]):
        sequence = []
        prev_index = None
        for t in range(predicted_indices[b].shape[0]):
            idx = int(predicted_indices[b][t])
            token = vocabulary[idx]
            if idx != blank_index and idx != prev_index and predicted_probs[b][t] > threshold:
                sequence.append([t, token, predicted_probs[b][t].item()])
                prev_index = idx
            else:
                prev_index = idx if idx != blank_index else prev_index
        batch_sequences.append(sequence)
    return batch_sequences

def median_filter_over_time(attention_maps, window_size):
    """
    Applies a median filter over the time dimension of attention maps.

    Parameters:
    - attention_maps (Tensor): Input tensor of shape [Batch, Time_length, Keywords].
    - window_size (int): The size of the median filter window (must be an odd integer).

    Returns:
    - Tensor: The filtered attention maps with the same shape as the input.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    pad_size = (window_size - 1) // 2
    # Permute the tensor to bring the time dimension to the last
    attention_maps_permuted = attention_maps.permute(0, 2, 1)  # Shape: [Batch, Keywords, Time_length]
    # Pad the time dimension (now the last dimension)
    padded_attention_maps = F.pad(
        attention_maps_permuted,
        pad=(pad_size, pad_size),  # Pad the last dimension (Time_length)
        mode='reflect'             # Options: 'reflect', 'replicate', 'constant'
    )
    # Unfold the time dimension to create sliding windows
    # The resulting shape will be [Batch, Keywords, Time_length, window_size]
    windows = padded_attention_maps.unfold(
        dimension=2,        # The time dimension (now the last dimension)
        size=window_size,   # Window size
        step=1              # Move one time step at a time
    )
    # Compute the median over the window dimension
    medians = windows.median(dim=3).values  # Shape: [Batch, Keywords, Time_length]
    # Permute back to the original shape
    medians = medians.permute(0, 2, 1)  # Shape: [Batch, Time_length, Keywords]
    return medians

def get_token_list(token_id_converter):
    """Retrieve token list from the token ID converter"""
    vocab_size = token_id_converter.get_num_vocabulary_size()
    return [token_id_converter.ids2tokens([i])[0] if len(token_id_converter.ids2tokens([i])) > 0 else '' for i in range(vocab_size)]

def map_token_to_context_index(token_id, biasing_list, token_list) -> int:
    """
    Naively maps a single token_id to a context index
    by checking if 'token_id' appears in any of the biasing_list entries.
    If not found, returns -1.
    
    NOTE: For multi-subword context items, you will likely need
          a more robust approach that checks entire subword sequences.
    """
    for c_idx, context_tokens in enumerate(biasing_list):
        # context_tokens might be the subword IDs for that particular context item
        if token_id in context_tokens:
            return c_idx
    return -1

def force_alignment(
    ctc_log_probs: torch.Tensor,
    target_tokens: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """
    Perform CTC forced alignment (Viterbi) between frame-wise CTC log-probs
    and a ground-truth token sequence.

    Args:
        ctc_log_probs (torch.Tensor): Shape (T, V). Log-probabilities from the CTC layer
            for each time frame T over vocabulary V.
        target_tokens (torch.Tensor): Shape (L,). Ground-truth label/token IDs.
        blank_id (int): Index for the CTC blank symbol.

    Returns:
        torch.Tensor: A 1D tensor of length T, containing the aligned token
            index (including blank) at each frame.
    """

    # -------------------------------------------------------------------------
    # 1) Expand the target to interleave blanks
    #    Example: target_tokens = [3, 5, 8]
    #    expanded_target = [blank, 3, blank, 5, blank, 8, blank]
    #    This is standard in CTC: We put a blank between each token (and at ends).
    # -------------------------------------------------------------------------
    L = target_tokens.size(0)
    expanded_length = 2 * L + 1
    expanded_tokens = []
    for i in range(L):
        expanded_tokens.append(blank_id)
        expanded_tokens.append(int(target_tokens[i]))
    expanded_tokens.append(blank_id)
    expanded_tokens = torch.tensor(expanded_tokens, device=ctc_log_probs.device, dtype=torch.long)
    # expanded_tokens: shape (2L+1,)

    # -------------------------------------------------------------------------
    # 2) Prepare DP matrices:
    #    alpha[t, i]: best log-likelihood of aligning up to frame t
    #                 with "i-th" position in the expanded target matched.
    #    backp[t, i]: which "i_prev" we came from in the previous time step.
    # -------------------------------------------------------------------------
    T, V = ctc_log_probs.size()
    alpha = torch.full((T, expanded_length), -math.inf, device=ctc_log_probs.device)
    backp = torch.full((T, expanded_length), -1, device=ctc_log_probs.device, dtype=torch.long)

    # For convenience:
    def logp(t, i):
        """Get log-prob for expanded token i at time t."""
        return ctc_log_probs[t, expanded_tokens[i]]

    # -------------------------------------------------------------------------
    # 3) Initialization at time t=0
    #    We can only match i=0 or i=1 at t=0:
    #     - i=0 must match the leading blank
    #     - i=1 could match the first real token (if we want an immediate label)
    # -------------------------------------------------------------------------
    alpha[0, 0] = logp(0, 0)  # blank alignment
    if expanded_length > 1:
        alpha[0, 1] = logp(0, 1)  # if we want to start matching the first token at t=0

    # -------------------------------------------------------------------------
    # 4) Viterbi recurrence
    #
    #   We define expanded_tokens: B T1 B T2 B T3 B ... T(L) B
    #
    #   Transitions:
    #   alpha[t, i] = logp(t, i) + max of:
    #       a) alpha[t-1, i]   (stay on same label, repeating blank or token)
    #       b) alpha[t-1, i-1] (move to next label, if i-1 >= 0)
    #       c) alpha[t-1, i-2] (skip over blank if i>1 and expanded_tokens[i] != expanded_tokens[i-2])
    # -------------------------------------------------------------------------
    for t in range(1, T):
        for i in range(expanded_length):
            # always can stay in the same expanded label (repeat)
            best_prev_val = alpha[t-1, i]
            best_prev_idx = i

            # can we move from i-1?
            if i - 1 >= 0:
                val = alpha[t-1, i-1]
                if val > best_prev_val:
                    best_prev_val = val
                    best_prev_idx = i-1

            # can we move from i-2? (CTC skip rule)
            # only if i>1 and the label at i != label at i-2
            if i - 2 >= 0:
                if expanded_tokens[i] != expanded_tokens[i-2]:
                    val = alpha[t-1, i-2]
                    if val > best_prev_val:
                        best_prev_val = val
                        best_prev_idx = i-2

            alpha[t, i] = logp(t, i) + best_prev_val
            backp[t, i] = best_prev_idx

    # -------------------------------------------------------------------------
    # 5) Find the best alignment score at final time T-1
    #    We pick whichever expanded label i has the best alpha[T-1, i].
    # -------------------------------------------------------------------------
    last_frame_best_i = torch.argmax(alpha[T-1])  # argmax over i in [0..2L]

    # -------------------------------------------------------------------------
    # 6) Backtrack to find the best path of "expanded label indices"
    #    We'll produce an array path_expanded of length T that tells
    #    which expanded label index i was chosen at each frame.
    # -------------------------------------------------------------------------
    path_expanded = torch.zeros((T,), dtype=torch.long, device=ctc_log_probs.device)
    path_expanded[T-1] = last_frame_best_i
    for t in reversed(range(1, T)):
        path_expanded[t-1] = backp[t, path_expanded[t]]

    # -------------------------------------------------------------------------
    # 7) Convert the "expanded label indices" to the actual token IDs
    #    i.e., alignment[t] = expanded_tokens[path_expanded[t]].
    # -------------------------------------------------------------------------
    alignment = expanded_tokens[path_expanded]

    return alignment

def is_suffix_subword(subword_str):
    # A naive check for a leading word boundary in sentencepiece or BPE
    # e.g. "▁" in sentencepiece indicates start of a new word.
    return subword_str.endswith("▁")

def get_word_boundaries(
    forced_align: torch.Tensor,
    tokens: torch.Tensor,
    token_list: list,
    blank_id: int = 0
):
    """
    Convert subword-level forced alignment into word-level start/end frames,
    but use a suffix-based rule to decide when a word ends.
    """
    # --------------------------------------------------
    # 1) Map each subword index in `tokens` to subword strings
    # --------------------------------------------------
    subword_seq = []
    for i, tok_id in enumerate(tokens):
        if tok_id.item() != blank_id:
            sub_tok_str = token_list[tok_id.item()]
            subword_seq.append((i, sub_tok_str))

    # --------------------------------------------------
    # 2) Build subword -> (start, end) frame from `forced_align`
    # --------------------------------------------------
    T = forced_align.size(0)
    subword_frame_map = dict()

    for t in range(T):
        tok_id = forced_align[t].item()
        if tok_id == blank_id:
            continue

        match_positions = (tokens == tok_id).nonzero(as_tuple=True)[0]
        if len(match_positions) == 0:
            continue

        found_i = None
        for i_ in match_positions:
            i_ = i_.item()
            if i_ not in subword_frame_map:
                found_i = i_
                break
            else:
                start_end = subword_frame_map[i_]
                if start_end[1] < 0:
                    found_i = i_
                    break
        if found_i is None:
            continue

        if found_i not in subword_frame_map:
            subword_frame_map[found_i] = [t, t]  # (start, end)
        else:
            subword_frame_map[found_i][1] = t

    # --------------------------------------------------
    # 3) Combine subwords into words using "suffix" rule
    # --------------------------------------------------
    word_boundaries = []
    current_word_pieces = []
    current_start = None
    current_end = None

    for i, sub_tok_str in subword_seq:
        if i not in subword_frame_map:
            # This subword had no aligned frames
            continue

        st, ed = subword_frame_map[i]  # subword-level alignment
        if current_word_pieces:
            # Extend the time boundary if needed
            if st < current_start:
                current_start = st
            if ed > current_end:
                current_end = ed
        else:
            # This is the first subword in the current "word" segment
            current_start = st
            current_end = ed

        current_word_pieces.append(sub_tok_str)

        # Check if this subword is a suffix subword => finalize
        if is_suffix_subword(sub_tok_str):
            # finalize the word
            word_str = "".join(current_word_pieces)
            # (Optional) remove any special tokens, e.g. '</w>'
            word_str = word_str.replace('</w>', '')
            # record boundary
            word_boundaries.append((word_str, current_start, current_end))

            # reset
            current_word_pieces = []
            current_start = None
            current_end = None

    # If something remains unfinished (e.g., no suffix subword at the end),
    # finalize it as well:
    if current_word_pieces:
        word_str = "".join(current_word_pieces).replace('</w>', '')
        word_boundaries.append((word_str, current_start, current_end))

    return word_boundaries


@torch.no_grad()
def forward(model, speech, speech_length, context_data, tokens, text, token_list):
    """
    Forward pass -> forced alignment -> context confusion matrix at the *word level*.

    We'll map each *word* in the reference to an optional context index,
    then pick a predicted context index for that entire word by averaging
    context probabilities across that word's frames.
    """
    # 1) Encoder forward
    encoder_output, encoder_output_lengths = model.encode(speech, speech_length)

    context_probabilities = None
    # 2) If using a context adapter on the encoder side
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
        encoder_bias_vector, encoder_attention = model.contextualizer(
            model_embed=encoder_output,
            context_embed=context_data["blist"],
            context_xphone_idxs=context_data["blist_xphone_mean"],
            ilens=context_data["ilens"],
            return_atten=True,
        )
        # encoder_attention shape: (B, #heads, T, #contexts)
        context_probabilities = torch.mean(encoder_attention, dim=1)  # (B=1, T, #contexts)
        encoder_output = encoder_output + encoder_bias_vector

    # 3) Compute CTC log-probs for forced alignment
    ctc_log_probs = F.log_softmax(model.ctc.ctc_lo(encoder_output), dim=-1)  # (B=1, T, vocab_size)

    # 4) Perform forced alignment (subword-level)
    forced_align = force_alignment(ctc_log_probs[0], tokens[0], blank_id=0)
    # forced_align: (T,)

    # 5) Convert subword alignment into *word-level* boundaries
    word_bound_list = get_word_boundaries(
        forced_align,
        tokens[0], 
        token_list,
        blank_id=0
    )
    # word_bound_list is a list of (word_str, start_frame, end_frame)

    # 6) Build confusion matrix: shape (#context_items+1, #context_items+1)
    #    The extra row/col is for "no context".
    confusion_matrix = None
    if context_probabilities is not None:
        T, num_contexts = context_probabilities.shape[1], context_probabilities.shape[2]
        confusion_matrix = {}

        # We'll build a mapping from *word_str* -> gold_context_index
        # or = num_contexts for "no context"
        # Suppose context_data["blist"] is a list of *words* that are considered context items.
        # Example: context_data["blist"] = ["apple", "banana", "opera"]...
        # Or it might be subwords. You may need to adapt accordingly.

        context_words = [w.lower() for w in context_data["context_list"]]  # e.g. to handle case
        def get_gold_context_index(word_str: str) -> int:
            w_lower = word_str.lower().replace('▁', '')
            if w_lower in context_words:
                return context_words.index(w_lower)
            else:
                return 0  # "no context"
            
        def get_gold_context(word_str: str) -> int:
            w_lower = word_str.lower().replace('▁', '')
            if w_lower in context_words:
                return w_lower.upper()
            else:
                return '<no-context>'

        # For each reference word:
        for (word_str, start_f, end_f) in word_bound_list:
            if start_f > end_f or start_f < 0 or end_f >= T:
                # invalid boundary, skip
                continue

            # 6(a) gold_context_idx
            gold_context = get_gold_context(word_str)

            # 6(b) predicted_context_idx by averaging context probs over [start_f, end_f]
            # shape of context_probabilities: (1, T, #contexts)
            slice_probs = context_probabilities[0, start_f:end_f+1, :]  # shape (end_f - start_f+1, #contexts)
            avg_probs = slice_probs.mean(dim=0)  # shape (#contexts,)
            pred_context= context_data['context_list'][int(avg_probs.argmax(dim=-1).item())]
            
            key = f'{gold_context}_{pred_context}'
            confusion_matrix[key] = 1 + confusion_matrix[key] if key in confusion_matrix else 1
    # print(f'confusion_matrix:\n{confusion_matrix}')
    return confusion_matrix

def main():
    # 1) Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # File paths
    spm_path = "./data/en_token_list/bpe_unigram5000suffix/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram5000suffix/tokens.txt"
    model_conf = "./conf/conformer/context_adapter.yaml"
    model_path = "./exp/asr_conformer/run_context_adapter_encoder_suffix/valid.acc.ave_10best.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe5000_sp_suffix/train/feats_lengths_stats.npz"

    rareword_path = "./local/contextual/contexts/context_f65536_train.txt"
    dataset_name = "S95_sp"
    speech_scp_path = f"./dump/raw/{dataset_name}/wav.scp"
    biasing_list_path = f"./dump/raw/{dataset_name}/uttblist_idx_f65536"
    biasing_list_xphone_path = None
    reference_path = f"./data/{dataset_name}/text"

    folder_name = model_path.split('/')[-1].split('.')[0]
    debug_path = os.path.join("/".join(model_path.split('/')[:-1]), 'debug_full', folder_name)
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # Load reference texts
    reference_texts = {d[0]: " ".join(d[1:]) for d in read_file(reference_path, sp=' ')}

    # Model loading
    data_path_and_name_and_type = [
        (speech_scp_path, 'speech', 'kaldi_ark'),
        (biasing_list_path, 'uttblist_idx', 'multi_columns_text'),
    ]
    contextual_conf = {
        'contextual_type': 'context_sampler',
        'context_list_path': rareword_path,
        'context_phone_embedding_path': biasing_list_xphone_path,
        'max_batch_disrupt_context': 21809,
        'sub_context_list_dropout': 0.0,
        'warmup_epoch': 0,
        'use_no_context_token': True,
        'context_prompt_has_context_template': '主題為:',
        'context_prompt_no_context_template': '開始吧',
    }

    model, loader, contextual_processor = load_espnet_model(
        model_conf=model_conf,
        contextual_conf=contextual_conf,
        token_path=token_path,
        context_token_path=token_path,
        frontend='default',
        stats_path=stats_path,
        spm_path=spm_path,
        context_spm_path=spm_path,
        model_path=model_path,
        data_path_and_name_and_type=data_path_and_name_and_type,
        return_contextual_processor=True,
        use_local_attn_conv=False,
        token_type='bpe',
        context_token_type='bpe',
    )

    # 2) Move model to device
    model.to(device)
    model.eval()

    # Prepare tokenizer and token list
    preprocessor = loader.dataset.preprocess
    tokenizer = preprocessor.tokenizer
    token_id_converter = preprocessor.token_id_converter
    token_list = get_token_list(token_id_converter) + ['<no-context>']

    golabel_confusion_matrix = {}
    count = 0

    for data_batch in tqdm(loader):
        uid = data_batch[0][0]
        if 'sp' in uid:
            continue
        if count >= 100000:
            break
        count += 1

        data = data_batch[1]
        context_data = data['contexts']

        # Move Tensors to device
        speech = data['speech'].to(device)              # (B, T, ...)
        speech_length = data['speech_lengths'].to(device)  # (B,)

        text = reference_texts[uid]
        biasing_list = context_data['blist']    # Possibly a list of lists?
        label_ctc = context_data['label_ctc']   # Possibly a tensor?

        # Convert text -> tokens and move to device
        text_tokens = preprocessor._text_process({'text': text})['text']
        tokens = torch.tensor(text_tokens).long().unsqueeze(0).to(device)

        # (Optional) Convert 'biasing_list' to CPU or GPU as needed
        # If it's a list of lists with integer Tensors, do something like:
        # for i in range(len(biasing_list)):
        #     biasing_list[i] = biasing_list[i].to(device)

        # For debugging, see if biasing_list is Tensors or just lists
        _biasing_list = [tokenizer.tokens2text([token_list[word] for word in rareword if word != -1]) 
                         for rareword in biasing_list]
        biasing_list = _biasing_list

        # Now run forward
        confusion_matrix = forward(model, speech, speech_length, context_data, tokens, text, token_list)

        # Aggregate
        if confusion_matrix is not None:
            for key, value in confusion_matrix.items():
                golabel_confusion_matrix[key] = golabel_confusion_matrix.get(key, 0) + value
    
        if count % 1000 == 0:
            output_path = os.path.join(debug_path, f'confusion_matrix_{dataset_name}_{count}.json')
            # sort the confusion matrix by counts then keys
            write_json(output_path, golabel_confusion_matrix)
            print(f'golabel_confusion_matrix:\n{golabel_confusion_matrix}')

    output_path = os.path.join(debug_path, f'confusion_matrix_{dataset_name}_final.json')
    # sort the confusion matrix by counts then keys
    golabel_confusion_matrix = dict(sorted(golabel_confusion_matrix.items(), key=lambda x: (-x[1], x[0])))
    write_json(output_path, golabel_confusion_matrix)
    print(f'golabel_confusion_matrix:\n{golabel_confusion_matrix}')

if __name__ == "__main__":
    main()