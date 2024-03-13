import os
import torch
import argparse
import numpy as np
import torchaudio
import sentencepiece as spm

from espnet2.asr.ctc                  import CTC
from espnet2.tasks.asr                import ASRTask
from espnet2.text.build_tokenizer     import build_tokenizer
from espnet2.text.token_id_converter  import TokenIDConverter
from espnet2.utils.get_default_kwargs import get_default_kwargs

from pyscripts.utils.fileio import read_yml

def load_espnet_model(
    model_conf, 
    contextual_conf, 
    token_path, 
    stats_path=None, 
    spm_path=None, 
    model_path='',
    data_path_and_name_and_type=None
):
    conf = read_yml(model_conf)
    conf['token_list']       = token_path
    conf['input_size']       = None if 'input_size' not in conf else conf['input_size']
    conf['specaug']          = None if 'specaug' not in conf else conf['specaug']
    conf['normalize']        = 'global_mvn' if 'normalize' not in conf else conf['normalize']
    conf['frontend']         = None
    conf['frontend_conf']    = {} if 'frontend_conf' not in conf else conf['frontend_conf']
    conf['ctc_conf']         = get_default_kwargs(CTC)
    conf['init']             = None
    conf['normalize_conf']   = {'stats_file': stats_path} if stats_path is not None else {}
    conf['token_type']       = 'bpe' if 'token_type' not in conf else conf['token_type']
    conf['bpemodel']         = spm_path
    conf['g2p']              = None if 'g2p' not in conf else conf['g2p']
    conf['cleaner']          = None if 'cleaner' not in conf else conf['cleaner']
    conf['use_preprocessor'] = True if 'use_preprocessor' not in conf else conf['use_preprocessor']
    conf['collate_fn_type']  = 'contextual' if 'collate_fn_type' not in conf else conf['collate_fn_type']

    conf['preprocessor']           = 'contextual' if 'preprocessor' not in conf else conf['preprocessor']
    conf['preprocessor_conf']      = {}
    conf['non_linguistic_symbols'] = None if 'non_linguistic_symbols' not in conf else conf['non_linguistic_symbols']

    conf['contextual_conf'].update(contextual_conf)

    args      = argparse.Namespace(**conf)
    # bpemodel  = spm.SentencePieceProcessor(model_file=spm_path)
    # tokenizer = build_tokenizer(token_type="bpe", bpemodel=spm_path)
    # converter = TokenIDConverter(token_list=args.token_list)
    
    # build model
    model = ASRTask.build_model(args)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # contextual processor
    # contextual_processor = ASRTask.build_contextual_processor(
    #     args, model
    # )
    
    # build loader
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        batch_size=1,
        num_workers=1,
        dtype='float32',
        preprocess_fn=ASRTask.build_preprocess_fn(args, False),
        collate_fn=ASRTask.build_collate_fn(args, False),
        allow_variable_data_keys=True,
        inference=True,
    )
    # return model, bpemodel, tokenizer, converter, loader
    return model, loader