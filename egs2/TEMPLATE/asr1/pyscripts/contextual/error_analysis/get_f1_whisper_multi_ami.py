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
from pyscripts.contextual.utils.visualize      import (
    plot_attention_map,
    plot_tsne,
    plot_gate
)
from espnet2.asr_transducer.utils import get_transducer_task_io

from espnet2.asr.contextualizer.func.contextual_adapter_func   import forward_contextual_adapter
from espnet.nets.pytorch_backend.transformer.add_sos_eos       import add_sos_eos
from espnet2.asr.contextualizer import (
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_token_list(token_id_converter):
    vocab_size = token_id_converter.get_num_vocabulary_size()
    vocab = []
    for i in range(vocab_size):
        v = token_id_converter.ids2tokens([i])
        v = v[0] if len(v) > 0 else ''
        vocab.append(v)
    return vocab

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
    encoder_out_original   = encoder_out

    # c1. Encoder contextualization
    atten = None
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
        bias_vec, atten = forward_contextual_adapter(
            contextualizer=model.contextualizer,
            model_embed=encoder_out,
            context_idxs=contexts['blist'],
            context_xphone_idxs=contexts['blist_xphone_mean'],
            ilens=contexts['ilens'],
            return_atten=True
        )
        atten       = atten.squeeze(1)
        # label_prior = torch.mean(atten, dim=1)
        # label_prior = torch.exp(0.3 * torch.log(label_prior))
        # atten       = (atten.squeeze(0) / label_prior).unsqueeze(0)
        encoder_out = encoder_out + bias_vec
        
        # contextual ctc decode
        context_ctc_idx = torch.argmax(atten, dim=-1).view(-1)
        context_idx     = torch.unique(context_ctc_idx)

    # ys_pad_lens = torch.tensor([d.shape[0] for d in tokens]).long()
    # ys_in_pad, ys_out_pad = add_sos_eos(
    #     tokens, model.sos, model.eos, model.ignore_id
    # )
    # ys_in_lens = ys_pad_lens + 1

    # 1. Forward decoder
    # outputs = model.decoder(
    #     encoder_out, enc_olens, ys_in_pad, ys_in_lens, return_hs=True
    # )
    # decoder_hs = outputs[0][1]
    # decoder_out = model.decoder.output_layer(decoder_hs)
    return context_idx.tolist()[1:]

if __name__ == "__main__":
    spm_path   = "whisper_multilingual"
    token_path = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/data/en_token_list/whisper_multilingual/tokens.txt"
    model_conf = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/conf/contextual_adapter/whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch.yaml"
    model_path = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/exp/asr_whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch/7epoch.pth"
    stats_path = None

    rare_path  = "/share/nas165/litingpai/datasets/ami/new_sorted_name.txt"
    scp_path   = "./dump/raw/ihm_eval/wav.scp"
    blist_path = "./dump/raw/ihm_eval/uttblist_idx"
    ref_path   = "./dump/raw/ihm_eval/text"

    folder_name = model_path.split('/')[-1].split('.')[0]
    debug_path = os.path.join("/".join(model_path.split('/')[:-1]), 'debug', folder_name)
    if not os.path.isdir(debug_path):
        os.makedirs(debug_path)

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    data_path_and_name_and_type = [
        (scp_path, 'speech', 'sound'), 
        (blist_path, 'uttblist_idx', 'multi_columns_text')
    ]

    contextual_conf = {
        'contextual_type': 'rareword',
        'blist_path': rare_path,
        'blist_xphone_path': '/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/contextual/ssl_features/new_sorted_name.xphone.seq.pt',
        'blist_max': 20,
        'blist_drop_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
        'use_oov': True,
    }
    
    model, loader = load_espnet_model(
        model_conf,
        contextual_conf, 
        token_path,
        None, 
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type,
        use_local_attn_conv=False,
        token_type='whisper_multilingual'
    )
    preprocessor       = loader.dataset.preprocess
    token_id_converter = preprocessor.token_id_converter
    token_list         = get_token_list(token_id_converter) + ['<oov>']

    model.eval()
    count = 0
    result = {}
    for data in tqdm(loader):
        # if count >= 1:
        #     break
        count += 1

        uid  = data[0][0]
        data = data[1]
        contexts       = data['contexts']
        speech         = data['speech']
        speech_lengths = data['speech_lengths']
        text           = texts[uid]
        label_ctc      = contexts['label_ctc'].tolist()[0]

        tokens = torch.tensor(
            preprocessor._text_process(
                {'text': text}
        )['text']).long()

        tokens = tokens.unsqueeze(0)
        context_idx = forward(
            model, 
            speech, 
            speech_lengths,
            contexts,
            tokens,
            text
        )
        # print(f'label: {label_ctc}, context_idx: {context_idx}')
        result[uid] = {
            'label': label_ctc,
            'pred' : context_idx
        }
    output_path = os.path.join(debug_path, 'predicts_ami.json')
    write_json(output_path, result)