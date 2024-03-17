import os
import torch
import random
import torchaudio
import numpy as np

from tqdm import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.contextual.utils.model          import load_espnet_model
from pyscripts.contextual.utils.rnnt_decode    import infernece
from pyscripts.contextual.utils.rnnt_alignment import forward_backward as force_alignment
from pyscripts.contextual.utils.visualize      import plot_attention_map

from espnet2.asr_transducer.utils import get_transducer_task_io

from espnet2.asr.contextualizer.func.contextual_adapter_func   import forward_contextual_adapter
from espnet.nets.pytorch_backend.transformer.add_sos_eos       import add_sos_eos
from espnet2.asr.contextualizer.func.contextualization_choices import (
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_token_list(token_id_converter):
    vocab_size = token_id_converter.get_num_vocabulary_size()
    vocab      = [
        token_id_converter.ids2tokens(
            [i], 
            # skip_special_tokens=False
        )[0] for i in range(vocab_size)
    ]
    return vocab

def visualize(
    logp,
    atten,
    text,
    target,
    blist,
    speech,
    blank_id, 
    token_list,
    debug_path,
    uttid,
):
    # alignments = force_alignment(
    #     logp, 
    #     target[0], 
    #     blank_id, 
    #     token_list,
    #     speech,
    #     debug_path
    # )
    # frame2align = {start: token for token, start, end in alignments}
    frame2align = {}

    plot_attention_map(
        frame2align,
        atten,
        text,
        blist,
        debug_path,
        uttid=uttid,
    )

@torch.no_grad()
def forward(
    model, 
    speech, 
    lengths,
    contexts,
    tokens,
    text
):
    encoder_out, enc_olens = model.encode(speech, lengths)
    print(f'encoder_out: {encoder_out.shape}')

    # c1. Encoder contextualization
    atten = None
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
        bias_vec, atten = forward_contextual_adapter(
            decoder=model.decoder,
            contextualizer=model.contextualizer,
            model_embed=encoder_out,
            context_idxs=contexts['blist'],
            ilens=contexts['ilens'],
            return_atten=True
        )
        atten = atten.squeeze(1)
        print(f'atten: {atten.shape}')
        encoder_out = encoder_out + bias_vec
    
    ys_pad_lens = torch.tensor([d.shape[0] for d in tokens]).long()
    print(ys_pad_lens)
    ys_in_pad, ys_out_pad = add_sos_eos(
        tokens, model.sos, model.eos, model.ignore_id
    )
    ys_in_lens = ys_pad_lens + 1

    # 1. Forward decoder
    outputs = model.decoder(
        encoder_out, enc_olens, ys_in_pad, ys_in_lens, return_hs=True
    )
    decoder_hs = outputs[0][1]

    # c1. Decoder contextualization
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_DECODER:
        print(f'Decoder contextualize!')
        bias_vec, atten = forward_contextual_adapter(
            decoder=model.decoder,
            contextualizer=model.contextualizer,
            model_embed=decoder_hs,
            context_idxs=contexts['blist'],
            ilens=contexts['ilens'],
            return_atten=True,
        )
        atten = atten.squeeze(1)
        print(f'atten: {atten.shape}')
        decoder_hs = decoder_hs + bias_vec

    decoder_out = model.decoder.output_layer(decoder_hs)
    logp        = None
    target      = None
    return logp, target, atten


if __name__ == "__main__":
    # spm_path   = "whisper_en"
    # token_path = "./data/en_token_list/whisper_en/tokens.txt"
    spm_path   = "./data/en_token_list/bpe_unigram600/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600/tokens.txt"
    model_conf = "./conf/exp/contextual_adapter/train_whisper_tiny_en_contextual_adapter_embed_tf_encoder_gactc.yaml"
    model_path = "./exp/asr_finetune_freeze_whisper_tiny_bpe600_cb_embed_tf_gactc_suffix/valid.loss.best.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_sp_suffix/train/feats_lengths_stats.npz"
    # stats_path = None
    
    rare_path  = "./local/contextual/rarewords/rareword_f15.txt"
    scp_path   = "./dump/raw/test_clean/wav.scp"
    blist_path = "./dump/raw/test_clean/uttblist_idx"
    ref_path   = "./data/test_clean/text"

    debug_path = os.path.join("/".join(model_path.split('/')[:-1]), 'debug')
    if not os.path.isdir(debug_path):
        os.mkdir(debug_path)

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    data_path_and_name_and_type = [
        (scp_path, 'speech', 'kaldi_ark'), 
        (blist_path, 'uttblist_idx', 'multi_columns_text')
    ]

    contextual_conf = {
        'contextual_type': 'rareword',
        'blist_path': './local/contextual/rarewords/all_rare_words.txt',
        'blist_max': 20,
        'blist_droup_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
    }
    
    model, loader = load_espnet_model(
        model_conf,
        contextual_conf, 
        token_path,
        None, 
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type
    )
    # print(model)
    preprocessor       = loader.dataset.preprocess
    token_id_converter = preprocessor.token_id_converter
    token_list         = get_token_list(token_id_converter) + ['<oov>']

    # print(token_list)

    model.eval()
    count = 0
    for data in loader:
        if count > 5:
            break
        count += 1

        uid  = data[0][0]
        data = data[1]
        contexts       = data['contexts']
        speech         = data['speech']
        speech_lengths = data['speech_lengths']
        text           = texts[uid]
        blist          = contexts['blist']
        ilens          = contexts['ilens']

        print(f'texts: {text}')
        print(f'uid: {uid}')
        print(f'blist:\n{blist}')
        print(f'ilens:\n{ilens}')
        print(f'speech: {speech}')
        print(f'speech_lengths: {speech_lengths}')
        
        _blist = []
        for rareword in blist:
            btokens = "".join([token_list[word] for word in rareword if word != -1])
            print(f'btokens: {btokens}, {rareword}')
            _blist.append(btokens)
        blist = _blist

        tokens = torch.tensor(
            preprocessor._text_process(
                {'text': text}
        )['text']).long()

        tokens = tokens.unsqueeze(0)
        # print(f'tokens : {tokens}')

        logp, target, atten = forward(
            model, 
            speech, 
            speech_lengths,
            contexts,
            tokens,
            text
        )

        visualize(
            logp,
            atten,
            text,
            target,
            blist,
            speech,
            model.blank_id, 
            token_list,
            debug_path,
            uid,
        )
        # break