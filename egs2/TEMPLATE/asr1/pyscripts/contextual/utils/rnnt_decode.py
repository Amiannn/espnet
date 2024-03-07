import os
import torch
import argparse
import numpy as np
import torchaudio
import sentencepiece as spm

from tqdm import tqdm

from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis

def greedy_search(model, enc_out):
    dec_state = model.decoder.init_state(1)

    hyp = Hypothesis(score=0.0, yseq=[model.blank_id], dec_state=dec_state)
    cache = {}

    dec_out, state, _ = model.decoder.score(hyp, cache)
    for enc_out_t in enc_out:
        
        lin_encoder_out = model.joint_network.lin_enc(enc_out_t)
        lin_decoder_out = model.joint_network.lin_dec(dec_out)
        # aco_bias        = model.get_acoustic_biasing_vector(enc_out_t, cb_embeds)
        # lin_encoder_out = lin_encoder_out + aco_bias

        joint_out = model.joint_network.joint_activation(
            lin_encoder_out + lin_decoder_out
        )
        joint_out = model.joint_network.lin_out(joint_out)
        logp      = torch.log_softmax(joint_out, dim=-1)
        top_logp, pred = torch.max(logp, dim=-1)

        if pred != model.blank_id:
            hyp.yseq.append(int(pred))
            hyp.score += float(top_logp)
            hyp.dec_state = state
            dec_out, state, _ = model.decoder.score(hyp, cache)
    return [hyp]

def decode_single_sample(model, tokenizer, converter, enc, nbest):
    nbest_hyps = greedy_search(model, enc)
    nbest_hyps = nbest_hyps[: nbest]

    results = []
    for hyp in nbest_hyps:
        assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

        # remove sos/eos and get results
        last_pos = None if model.use_transducer_decoder else -1
        if isinstance(hyp.yseq, list):
            token_int = hyp.yseq[1:last_pos]
        else:
            token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        # Change integer-ids to tokens
        token = converter.ids2tokens(token_int)

        if tokenizer is not None:
            text = tokenizer.tokens2text(token)
        else:
            text = None
        results.append((text, token, token_int, hyp))
    return results

@torch.no_grad()
def infernece(
    model,
    tokenizer,
    converter,
    speech, 
    device='cpu'
):
    """Inference

    Args:
        data: Input speech data
    Returns:
        text, token, token_int, hyp

    """
    # data: (Nsamples,) -> (1, Nsamples)
    speech = speech.unsqueeze(0)
    # lengths: (1,)
    lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    batch = {"speech": speech, "speech_lengths": lengths}
    print("speech length: " + str(speech.size(1)))

    # b. Forward Encoder
    enc, enc_olens = model.encode(**batch)
    
    # Normal ASR
    intermediate_outs = None
    if isinstance(enc, tuple):
        intermediate_outs = enc[1]
        enc = enc[0]
    assert len(enc) == 1, len(enc)

    # c. Passed the encoder result and the beam search
    results = decode_single_sample(
        model, tokenizer, converter, enc[0], nbest=1,
    )
    return results