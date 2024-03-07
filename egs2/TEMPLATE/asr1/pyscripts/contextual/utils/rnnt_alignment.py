import os
import json
import torch
import logging
import numpy as np

import math  
import textgrid
import matplotlib.pyplot as plt

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_file

from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union
)

from espnet2.tasks.asr            import ASRTask
from espnet2.bin.asr_inference    import Speech2Text
from espnet2.asr_transducer.utils import get_transducer_task_io

from espnet2.torch_utils.set_all_random_seed             import set_all_random_seed
from espnet2.asr.transducer.beam_search_transducer       import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer       import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

def get_vertical_transition(logp, target):
    u_len, t_len = logp.shape[:2]

    u_idx = torch.arange(0, u_len - 1, dtype=torch.long).repeat(t_len, 1).T.reshape(-1)
    t_idx = torch.arange(0, t_len, dtype=torch.long).repeat(u_len - 1)
    y_idx = target.repeat(t_len, 1).T.reshape(-1).long()

    y_logp = torch.zeros(u_len, t_len)
    y_logp[:-1, :] = logp[u_idx, t_idx, y_idx].reshape(u_len - 1, t_len)
    y_logp[-1:, :] = -float("inf")
    return y_logp

def get_horizontal_transition(logp, target, blank_id):
    u_len, t_len = logp.shape[:2]

    u_idx   = torch.arange(0, u_len, dtype=torch.long).repeat(t_len - 1, 1).T.reshape(-1)
    t_idx   = torch.arange(0, t_len - 1, dtype=torch.long).repeat(u_len)
    phi_idx = torch.zeros(u_len * (t_len - 1), dtype=torch.long) + blank_id

    phi_logp = torch.zeros(u_len, t_len)
    phi_logp[:, :-1] = logp[u_idx, t_idx, phi_idx].reshape(u_len, t_len - 1)
    phi_logp[:, -1:] = -float("inf")
    return phi_logp

@torch.no_grad()
def forward_backward(logp, target, blank_id, token_list, waveform, debug_path):
    u_len, t_len = logp.shape[:2]

    y_logp   = get_vertical_transition(logp, target)
    phi_logp = get_horizontal_transition(logp, target, blank_id)

    alpha = torch.zeros(u_len, t_len)
    zero_tensor = torch.zeros(1)
    inf_tensor  = torch.zeros(1) + -float("inf")
    for u in range(u_len):
        for t in range(t_len):
            if u == 0 and t == 0: continue
            alpha_y_partial   = alpha[u - 1, t] + y_logp[u - 1, t] if (u - 1) >= 0 else inf_tensor
            alpha_phi_partial = alpha[u, t - 1] + phi_logp[u, t - 1] if (t - 1) >= 0 else inf_tensor
            alpha[u, t] = torch.logaddexp(alpha_y_partial, alpha_phi_partial)
    plt.imshow(alpha, origin="lower")
    output_path = os.path.join(debug_path, 'alpha.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    beta = torch.zeros(u_len, t_len)
    for u in range(u_len - 1, -1, -1):
        for t in range(t_len - 1, -1, -1):
            if u == (u_len - 1) and t == (t_len - 1): continue
            beta_y_partial   = beta[u + 1, t] + y_logp[u, t] if (u + 1) < u_len else inf_tensor
            beta_phi_partial = beta[u, t + 1] + phi_logp[u, t] if (t + 1) < t_len else inf_tensor
            beta[u, t] = torch.logaddexp(beta_y_partial, beta_phi_partial)

    plt.imshow(beta, origin="lower")
    output_path = os.path.join(debug_path, 'beta.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    ab_log_prob = (alpha + beta)
    ab_prob     = torch.exp(ab_log_prob)

    plt.imshow(ab_log_prob, origin="lower")
    output_path = os.path.join(debug_path, 'ab_log_prob.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    plt.imshow(ab_prob, origin="lower")
    output_path = os.path.join(debug_path, 'ab.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    last_token  = 0
    align_paths = []
    align_path  = []
    target      = [0] + target.tolist()
    tokens      = [token_list[t] for t in target]
    
    now_u, now_t = u_len - 1, t_len - 1
    while (now_u >= 0 and now_t >= 0):
        if ab_prob[now_u, now_t - 1] > ab_prob[now_u - 1, now_t]:
            # stay
            now_t -= 1
            align_path = [now_t] + align_path
        else:
            # leave
            now_u -= 1
            if len(align_path) == 0:
                align_path = [now_t]
            align_paths = [align_path] + align_paths
            align_path = []

    fig, [ax1, ax2] = plt.subplots(
        2, 1,
        # figsize=(50, 10)
    )
    ax1.imshow(ab_prob, origin="lower", aspect='auto', interpolation='none')
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    sample_rate = 16000
    # The original waveform
    ratio = waveform.size(0) / sample_rate / t_len
    maxTime = math.ceil(waveform.size(0) / sample_rate)
    alignments = []
    tg = textgrid.TextGrid(minTime=0, maxTime=maxTime)
    tier_word = textgrid.IntervalTier(name="subword", minTime=0., maxTime=maxTime)
    for i, align_path in enumerate(align_paths):
        token = tokens[i]
        if len(align_path) == 0:
            continue
        start, end = min(align_path), max(align_path)
        if start == end:
            continue
        alignments.append([token, start, end])
        start    = ratio * start
        end      = ratio * end
        interval = textgrid.Interval(minTime=start, maxTime=end, mark=token)
        tier_word.addInterval(interval)
        start = f'{start:.2f}'
        end   = f'{end:.2f}'
    tg.tiers.append(tier_word)
    tg.write("1.TextGrid")
    ax2.specgram(waveform, Fs=sample_rate)
    ax2.set_yticks([])
    ax2.set_xlabel("time [second]")
    fig.tight_layout()
    output_path = os.path.join(debug_path, 'ab_prob.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
    return alignments