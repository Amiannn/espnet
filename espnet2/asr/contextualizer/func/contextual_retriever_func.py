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
        for idx in y_hat:
            idx = int(idx)
            token = token_list[int(idx)]
            if idx != -1 and idx != idx_blank and token not in seq_hat:
                seq_hat.append(token)
    hyp = []
    for i, h in enumerate(seq_hat):
        if i < (len(seq_hat) - 1):
            h = h + sep_tokens
        hyp.extend(h)
    return hyp