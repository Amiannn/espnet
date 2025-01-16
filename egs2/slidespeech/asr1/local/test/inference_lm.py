#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import sentencepiece as spm

from espnet2.tasks.lm import LMTask
from espnet2.torch_utils.device_funcs import to_device

def get_log_prob(text):
    tokens = [model.sos] + tokenizer.encode(text.upper())
    x = torch.tensor(tokens).unsqueeze(0)

    output, _ = model.lm(x, None)
    output = torch.log_softmax(output.squeeze(0), dim=-1)

    log_prob = torch.sum(output)
    return log_prob

if __name__ == "__main__":
    train_config = "./exp/lm/source_domain/config.yaml"
    model_file   = "./exp/lm/source_domain/1000epoch.pth"
    device       = "cpu"

    bpe_path = "./data/en_token_list/bpe_unigram5000suffix/bpe.model"
    tokenizer = spm.SentencePieceProcessor(model_file=bpe_path)

    model, train_args = LMTask.build_model_from_file(train_config, model_file, device)
    model.eval()
    print(model)

    text_a = "Today is a"
    log_prob_a = get_log_prob(text_a)

    text_b = "sdf dsfe fwe"
    log_prob_b = get_log_prob(text_b)

    print(f"Log prob of '{text_a}': {log_prob_a}")
    print(f"Log prob of '{text_b}': {log_prob_b}")

    tokens = [model.sos] + tokenizer.encode(text_a.upper())
    for i in range(20):
        x = torch.tensor(tokens).unsqueeze(0)
        x_len = torch.tensor([len(tokens)])

        output, _ = model.lm(x, None)
        output = output.squeeze(0)
        # print(output.shape)

        pred_tokens = output.argmax(dim=-1).squeeze(0).tolist()
        tokens = tokens + [pred_tokens[-1]]
        # for inp_token, pred_token in zip(tokens, pred_tokens):
        #     print(f"{inp_token} -> {pred_token}")
        print(tokenizer.decode(tokens[1:]))
        if pred_tokens[-1] == model.eos:
            break