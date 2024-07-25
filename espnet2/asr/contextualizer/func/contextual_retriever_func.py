import torch
import logging

from itertools import groupby

from espnet2.asr.decoder.whisper_decoder    import OpenAIWhisperDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

def retrieve_greedy_decode(probs, token_list, sep_tokens, idx_blank=0):
    ys_hat  = probs.argmax(dim=-1).cpu()
    for _, y in enumerate(ys_hat):
        y_hat = [x[0] for x in groupby(y)]
        seq_hat = []
        seq_idx = []
        for idx in y_hat:
            idx = int(idx)
            if idx != -1 and idx != idx_blank:
                seq_hat.append(token_list[int(idx)])
                seq_idx.append(int(idx))
    hyp     = []
    hyp_idx = []
    for i, h in enumerate(seq_hat):
        if seq_idx[i] in hyp_idx:
            continue
        if i < (len(seq_hat) - 1):
            h = h + sep_tokens
        hyp.extend(h)
        hyp_idx.append(seq_idx[i])
    return hyp