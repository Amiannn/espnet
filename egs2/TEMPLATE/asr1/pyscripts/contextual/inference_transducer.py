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
from espnet2.asr.contextualizer.func.contextualization_choices import (
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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
):
    alignments = force_alignment(
        logp, 
        target[0], 
        blank_id, 
        token_list,
        speech,
        debug_path
    )
    frame2align = {start: token for token, start, end in alignments}

    plot_attention_map(
        frame2align,
        atten,
        text,
        blist,
        debug_path,
        uttid='test',
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
    atten    = None
    bias_vec = None
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
        bias_vec, atten = forward_contextual_adapter(
            decoder=model.decoder,
            contextualizer=model.contextualizer,
            model_embed=encoder_out,
            context_idxs=contexts['blist'],
            context_xphone_idxs=contexts['blist_xphone'] if 'blist_xphone' in contexts else None,
            ilens=contexts['ilens'],
            return_atten=True
        )
        atten = atten.squeeze(1)
        print(f'atten: {atten.shape}')

    decoder_in, target, t_len, u_len = get_transducer_task_io(
        tokens,
        enc_olens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )
    decoder_out = model.decoder(decoder_in)
    print(f'decoder_out: {decoder_out.shape}')

    joint_out = model.joint_network(
        encoder_out.unsqueeze(2), 
        decoder_out.unsqueeze(1),
        bias_out=bias_vec.unsqueeze(2),
    )
    print(f'joint_out: {joint_out.shape}')
    logp = torch.log_softmax(joint_out, dim=-1)[0].transpose(1, 0)
    print(f'logp: {logp.shape}')

    return logp, target, atten


if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600/tokens.txt"
    model_conf = "./conf/exp/contextual_adapter/train_rnnt_contextual_adapter_tf_xphone_encoder_with_gactc.yaml"
    model_path = "./exp/asr_finetune_freeze_ct_enc_cb_tf_xphone_gactc_suffix/1epoch.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_sp_suffix/train/feats_lengths_stats.npz"
    
    rare_path  = "./local/contextual/rarewords/all_rare_words.txt"
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
        'blist_path': rare_path,
        'blist_xphone_path': './local/contextual/ssl_features/all_rare_words.xphone.pt',
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
        'default',
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type
    )

    preprocessor       = loader.dataset.preprocess
    token_id_converter = preprocessor.token_id_converter
    token_list         = token_id_converter.token_list + ['<oov>']
    print(f'token_list: {len(token_list)}')
    
    model.eval()
    for data in loader:
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
            print(f'btokens: {btokens}')
            _blist.append(btokens)
        blist = _blist

        tokens = torch.tensor(
            preprocessor._text_process(
                {'text': text}
        )['text']).long()

        tokens = tokens.unsqueeze(0)
        print(f'tokens : {tokens}')

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
        )
        break