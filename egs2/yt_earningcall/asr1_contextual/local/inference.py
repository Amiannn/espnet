import os
import torch
import argparse
import numpy as np
import torchaudio
import sentencepiece as spm

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file
from local.utils import read_yml

from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union
)

from espnet2.asr.ctc import CTC
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis

from espnet2.tasks.asr          import ASRTask
from espnet2.train.preprocessor import CommonPreprocessor

from espnet2.text.build_tokenizer     import build_tokenizer
from espnet2.text.token_id_converter  import TokenIDConverter
from espnet2.utils.get_default_kwargs import get_default_kwargs

def greedy_search(asr_model, enc_out):
    dec_state = asr_model.decoder.init_state(1)

    hyp = Hypothesis(score=0.0, yseq=[asr_model.blank_id], dec_state=dec_state)
    cache = {}

    dec_out, state, _ = asr_model.decoder.score(hyp, cache)
    for enc_out_t in enc_out:
        logp = torch.log_softmax(
            asr_model.joint_network(enc_out_t, dec_out),
            dim=-1,
        )
        top_logp, pred = torch.max(logp, dim=-1)

        if pred != asr_model.blank_id:
            hyp.yseq.append(int(pred))
            hyp.score += float(top_logp)
            hyp.dec_state = state
            dec_out, state, _ = asr_model.decoder.score(hyp, cache)
    return [hyp]

def decode_single_sample(asr_model, tokenizer, converter, enc, nbest):
    nbest_hyps = greedy_search(asr_model, enc)
    nbest_hyps = nbest_hyps[: nbest]

    results = []
    for hyp in nbest_hyps:
        assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

        # remove sos/eos and get results
        last_pos = None if asr_model.use_transducer_decoder else -1
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
def forward(
    asr_model,
    tokenizer,
    converter,
    speech, 
    device
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
    enc, enc_olens = asr_model.encode(**batch)
    
    # Normal ASR
    intermediate_outs = None
    if isinstance(enc, tuple):
        intermediate_outs = enc[1]
        enc = enc[0]
    assert len(enc) == 1, len(enc)

    # c. Passed the encoder result and the beam search
    results = decode_single_sample(
        asr_model, tokenizer, converter, enc[0], nbest=1,
    )
    return results

if __name__ == "__main__":
    spm_path   = "./data/token_list/bpe_unigram5000suffix/bpe.model"
    token_path = "./data/token_list/bpe_unigram5000suffix/tokens.txt"
    model_conf = "./conf/exp/train_asr_transducer_conformer.yaml"
    model_path = "./exp/asr_train_asr_transducer_conformer_raw_bpe5000_use_wandbtrue_sp_suffix/valid.loss.best.pth"
    stats_path = "./exp/asr_stats_raw_bpe5000_spsuffix/train/feats_stats.npz"
    
    conf = read_yml(model_conf)
    conf['token_list']     = token_path
    conf['input_size']     = None
    conf['specaug']        = None
    conf['normalize']      = 'global_mvn'
    conf['frontend']       = 'default'
    conf['ctc_conf']       = get_default_kwargs(CTC)
    conf['init']           = None
    conf['normalize_conf'] = {
        'stats_file': stats_path
    }

    args = argparse.Namespace(**conf)

    bpemodel  = spm.SentencePieceProcessor(model_file=spm_path)
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=spm_path)
    converter = TokenIDConverter(token_list=args.token_list)
    
    model = ASRTask.build_model(args)
    model.load_state_dict(torch.load(model_path))  

    audio_path = "./local/test.wav"
    audio_path = "/share/nas165/amian/experiments/speech/preprocess/dataset_pre/egs2/esun_investor_various/data_ws/wav_segment/esun2016Q3_187.wav"
    ref_text   = ""

    speech, sample_rate = torchaudio.load(audio_path)
    speech = speech.reshape(-1)
    
    results = forward(
        model,
        tokenizer,
        converter,
        speech, 
        device='cpu'
    )[0][0]

    print(f'Trans: {results}')
