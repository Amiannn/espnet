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
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER,
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

def visualize(
    logp,
    atten,
    enc_out,
    pred_ctc,
    gate_prob,
    text,
    target,
    blist,
    speech,
    blank_id, 
    token_list,
    debug_path,
    uttid,
):
    frame2align = {}

    plot_attention_map(
        frame2align,
        atten,
        text,
        blist,
        debug_path,
        uttid=uttid,
    )

    # pred_ctc = [token_list[p] if p != 0 else ' ' for p in pred_ctc]
    # plot_tsne(
    #     enc_out.squeeze(0),
    #     pred_ctc,
    #     debug_path,
    #     uttid=uttid,
    # )
    
    if gate_prob is not None:
        plot_gate(
            gate_prob,
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
    encoder_out_original   = encoder_out
    print(f'encoder_out: {encoder_out.shape}')

    ctc_pred = None
    if model.ctc is not None:
        out      = model.ctc.ctc_lo(encoder_out).squeeze(0)
        ctc_pred = torch.argmax(out, dim=-1)

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
        label_prior = torch.mean(atten, dim=1)
        label_prior = torch.exp(0.3 * torch.log(label_prior))
        atten       = (atten.squeeze(0) / label_prior).unsqueeze(0)
        encoder_out = encoder_out + bias_vec
        
        # contextual ctc decode
        context_ctc_idx = torch.argmax(atten, dim=-1).view(-1)
        contecxt_idx    = torch.unique(context_ctc_idx)
        print(f'contecxt_idx: {contecxt_idx}')

        gate_prob   = None
        if hasattr(model.contextualizer, "gate_prob"):
            gate_prob = model.contextualizer.gate_prob
            gate_prob = gate_prob.view(-1)
            # print(f'gate_value:\n{gate_prob}')

        # encoder_out_original = model.contextualizer.adapter.query_conv(
        #     encoder_out.transpose(1, 2)
        # ).transpose(1, 2)
        # print(f'encoder_out_original query: {encoder_out_original.shape}')

    ys_pad_lens = torch.tensor([d.shape[0] for d in tokens]).long()
    print(ys_pad_lens)
    ys_in_pad, ys_out_pad = add_sos_eos(
        tokens, model.sos, model.eos, model.ignore_id
    )
    ys_in_lens = ys_pad_lens + 1

    logp   = None
    target = None
    if model.decoder is None:
        return logp, target, atten, encoder_out_original, ctc_pred

    # 1. Forward decoder
    outputs = model.decoder(
        encoder_out, enc_olens, ys_in_pad, ys_in_lens, return_hs=True
    )
    decoder_hs = outputs[0][1]
    decoder_out = model.decoder.output_layer(decoder_hs)
    return logp, target, atten, encoder_out_original, ctc_pred, gate_prob

prompt_tokens = [50259, 41847, 40925,  2532, 11944,  8232, 38391,     6,    50, 37173, 36880,  6205,  8726,    37, 10253,    45, 12268,    11,  7469, 17188, 25240, 20047,    11,  6999, 12243, 22528,    11, 18050,    38,  4731, 11,  3002, 12224, 11435, 29398,    11,  9872, 24010, 14364, 12268, 13,  2264,  4299,  3578,  2195,   286,     6, 24010, 16596, 13758, 16309,    13, 50363, 15456,   307,   707,   295,   220,  3322,  6258, 563,  9072,   293,   862, 26872,   597,   321,   366, 27524,   220, 1353, 14644,   365,   220,  3322, 45328, 40008,   412,  1151,   220, 3322,  3303,   665,  3687,   307,  1257,  3004,   365, 20100,   420, 575,  3105,   666,   459, 19191,  1287,   293,  3114,   298]

if __name__ == "__main__":
    spm_path   = "whisper_multilingual"
    token_path = "./data/en_token_list/whisper_multilingual/tokens.txt"
    model_conf = "./conf/contextual_adapter/whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch.yaml"
    model_path = "./exp/asr_whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch/7epoch.pth"
    stats_path = None

    rare_path  = "./local/contextual/rarewords/all_rare_words.txt"
    scp_path   = "./dump/raw/test_clean/wav.scp"
    blist_path = "./dump/raw/test_clean/uttblist_idx"
    ref_path   = "./data/test_clean/text"

    folder_name = model_path.split('/')[-1].split('.')[0]
    debug_path = os.path.join("/".join(model_path.split('/')[:-1]), 'debug', folder_name)
    if not os.path.isdir(debug_path):
        os.makedirs(debug_path)

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    data_path_and_name_and_type = [
        (scp_path, 'speech', 'kaldi_ark'), 
        (blist_path, 'uttblist_idx', 'multi_columns_text')
    ]

    contextual_conf = {
        'contextual_type': 'rareword',
        'blist_path': rare_path,
        'blist_xphone_path': './local/contextual/ssl_features/all_rare_words.xphone.seq.pt',
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
    # print(model)
    preprocessor       = loader.dataset.preprocess
    token_id_converter = preprocessor.token_id_converter
    token_list         = get_token_list(token_id_converter) + ['<oov>']

    prompt = [token_list[idx] for idx in prompt_tokens]
    print(f'prompt: {prompt}')

    # model.contextualizer.adapter.temperature = 1

    # print(token_list)
    model.eval()
    count = 0
    for data in loader:
        if count >= 1:
            break
        count += 1

        uid  = data[0][0]
        data = data[1]
        contexts       = data['contexts']
        speech         = data['speech']
        speech_lengths = data['speech_lengths']
        text           = texts[uid]
        blist          = contexts['blist']
        # blist_xphone   = contexts['blist_xphone']
        ilens          = contexts['ilens']
        label_ctc      = contexts['label_ctc']
        blist_idxs     = contexts['blist_idxs']
        prompt         = data['prompt']

        print(f'texts: {text}')
        # print(f'uid: {uid}')
        # print(f'blist:\n{blist}')
        # print(f'blist_xphone:\n{blist_xphone.shape}')
        # print(f'ilens:\n{ilens}')
        # print(f'speech: {speech}')
        # print(f'speech_lengths: {speech_lengths}')
        print(f'label_ctc:\n{label_ctc}')
        # print(f'blist_idx:\n{blist_idxs}')
        print(f'prompt: {prompt}')

        _blist = []
        for rareword in blist:
            btokens = "".join([token_list[word] for word in rareword if word != -1])
            # print(f'btokens: {btokens}, {rareword}')
            _blist.append(btokens)
        blist = _blist

        tokens = torch.tensor(
            preprocessor._text_process(
                {'text': text}
        )['text']).long()

        tokens = tokens.unsqueeze(0)
        # print(f'tokens : {tokens}')

        logp, target, atten, enc_out, pred_ctc, gate_prob = forward(
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
            enc_out,
            pred_ctc,
            gate_prob,
            text,
            target,
            blist,
            speech,
            model.blank_id, 
            token_list,
            debug_path,
            uid,
        )