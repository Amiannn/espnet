import torch
import logging

from itertools import groupby

from espnet2.asr.decoder.whisper_decoder    import OpenAIWhisperDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

def retrieve_ctc_decode(
    probs,
    token_list,
    idx_blank=0,
    **kwargs,
):
    ys_hat = probs.argmax(dim=-1).cpu()
    for _, y in enumerate(ys_hat):
        y_hat = [x[0] for x in groupby(y)]
        seq_hat = []
        for idx in y_hat:
            idx = int(idx)
            token = token_list[int(idx)]
            if idx != -1 and idx != idx_blank and token not in seq_hat:
                seq_hat.append(token)
    return seq_hat

def topk_decode(
    probs, 
    token_list, 
    idx_blank=0, 
    top_k=10, 
    threshold=0.5
):
    ys_max       = probs.cpu().max(dim=-1)
    value_tensor = ys_max.values[0]
    index_tensor = ys_max.indices[0]
    
    target = torch.zeros(len(token_list))
    count  = torch.zeros(len(token_list))
    
    target.scatter_add_(0, index_tensor, value_tensor)
    count.scatter_add_(0, index_tensor, torch.ones_like(value_tensor))
    
    # normalize
    target = target / count
    target[count == 0] = 0
    target[idx_blank]  = 0

    indexis = torch.argsort(target, descending=True)[:top_k]
    indexis = [i for i in indexis if target[i] >= threshold]
    result  = [[i.item(), token_list[i], target[i].item()] for i in indexis]
    return result

def create_prompt(pred_tokens, sep_tokens, end_tokens):
    hyp = []
    for i, pred in enumerate(pred_tokens):
        _, h, _ = pred
        if i < (len(pred_tokens) - 1):
            h = h + sep_tokens
        else:
            h = h + end_tokens
        hyp.extend(h)
    return hyp