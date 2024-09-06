import os
import torch
import random
import numpy as np

from tqdm      import tqdm
from itertools import groupby

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

from espnet.nets.pytorch_backend.transformer.add_sos_eos     import add_sos_eos
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

from espnet2.asr.contextualizer.func.contextual_retriever_func import topk_decode
from espnet2.asr.contextualizer.func.contextual_retriever_func import retrieve_ctc_decode

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

import torch.nn.functional as F

def median_filter_tensor(input_tensor, kernel_size):
    # Assuming input_tensor is of shape B x T x C
    B, T, C = input_tensor.shape
    
    # Padding to handle borders
    pad = kernel_size // 2
    input_tensor_padded = torch.nn.functional.pad(input_tensor, (0, 0, pad, pad), mode='reflect')
    
    # Unfold the T dimension to create overlapping windows of size kernel_size
    unfolded = input_tensor_padded.unfold(1, kernel_size, 1)  # Shape: B x (T + 2*pad - kernel_size + 1) x kernel_size x C
    
    # Compute the median across the kernel_size dimension (third dimension)
    median_filtered = torch.median(unfolded, dim=2).values  # Shape: B x T x C
    
    return median_filtered

def get_token_list(token_id_converter):
    vocab_size = token_id_converter.get_num_vocabulary_size()
    vocab = []
    for i in range(vocab_size):
        v = token_id_converter.ids2tokens([i])
        v = v[0] if len(v) > 0 else ''
        vocab.append(v)
    return vocab

def retriever_decode(ys_hat, char_list, idx_blank=0):
    for _, y in enumerate(ys_hat):
        y_hat = [x[0] for x in groupby(y)]
        seq_hat = []
        for idx in y_hat:
            idx = int(idx)
            if idx != -1 and idx != idx_blank:
                seq_hat.append(char_list[int(idx)])
    hyp  = []
    for h in seq_hat:
        if h in hyp:
            continue
        hyp.append(h)
    return ", ".join(hyp)

def visualize(
    logp,
    atten,
    ctc_pred,
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

    if ctc_pred is not None:
        pred_ctc = [token_list[p] if p != 0 else ' ' for p in ctc_pred]
        frame2align = {i: pred_ctc[i] for i in range(atten.shape[1])}
        print(f'pred_ctc: {pred_ctc}')

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
    text,
    token_list,
):
    encoder_out, enc_olens = model.encode(speech, lengths)
    print(f'encoder_out: {encoder_out.shape}')

    # c1. Encoder contextualization
    encoder_proj = None
    if model.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_RETRIEVER:
        context_prob, encoder_proj = model.contextualizer(
            model_embed=encoder_out,
            context_embed=contexts['blist'],
            context_xphone_embed=contexts['blist_xphone'],
            context_xphone_mean_embed=contexts['blist_xphone_mean'],
            ilens=contexts['ilens'],
            return_model_proj=True
        )
        # medium filter
        # context_prob     = median_filter_tensor(context_prob, kernel_size=3)
        # print(f'- context_prob: {context_prob.shape}')
        # context_prob_sw  = model.contextualizer.retriever.maxsim_sw_prob
        # context_prob_pho = model.contextualizer.retriever.maxsim_pho_prob

        context_prob_sw  = torch.softmax(model.contextualizer.retriever.sw_score, dim=-1)
        context_prob_pho = torch.softmax(model.contextualizer.retriever.ph_score, dim=-1)

        print(f'context_prob: {context_prob.shape}')
        print(f'context_prob_sw: {context_prob_sw.shape}')
        print(f'context_prob_pho: {context_prob_pho.shape}')

        predict = topk_decode(
            context_prob, 
            blist, 
            idx_blank=0, 
            top_k=5, 
            threshold=0.9
        )
        # predict_ctc = retrieve_ctc_decode(
        #     context_prob,
        #     blist,
        #     idx_blank=0,
        #     threshold=0.6,
        # )
        print(f'Retriever predict: {predict}')
    
    ctc_pred = None
    if model.ctc is not None:
        print(f'encoder_proj: {encoder_proj.shape}')
        x        = encoder_proj if encoder_proj is not None else encoder_out
        ctc_pred = model.ctc.argmax(x).squeeze(0)
        print(f'ctc_pred: {ctc_pred.shape}')
    
    predict_hyp = retrieve_ctc_decode(
        model.ctc.ctc_lo(x),
        token_list,
        idx_blank=0,
        threshold=0.0,
    )
    predict_hyp = "".join([d[1] for d in predict_hyp]).replace("▁", ' ')
    predict = {
        'text'      : text,
        'hyp'       : predict_hyp,
        'result'    : predict,
        # 'result_ctc': predict_ctc,
    }
    logp        = None
    target      = None
    return logp, target, [context_prob, context_prob_sw, context_prob_pho], ctc_pred, predict

if __name__ == "__main__":
    spm_path   = "./data/token_list/bpe_unigram5000suffix/bpe.model"
    token_path = "./data/token_list/bpe_unigram5000suffix/tokens.txt"
    model_conf = "./conf/contextual/whisper/train_asr_whisper_medium_conformer_multilateinteractive_retrieval_balanced_alpha0.5_annhnw.yaml"
    model_path = "./exp/asr_whisper/run_medium_conformer_multilateinteractive_retrieval_balanced_alpha0.5_annhnw_suffix/valid.loss.ave_10best.pth"
    stats_path = "./exp/asr_stats_raw_bpe5000_sp_suffix/train/feats_lengths_stats.npz"

    # E.SUN
    # rare_path  = "./local/contextual/rarewords/rareword_f10_test.txt"
    rare_path  = "./local/contextual/rarewords/esun.entity.txt"
    scp_path   = "./dump/raw/test/wav.scp"
    # blist_path = "./dump/raw/test/uttblist_idx"
    blist_path = "./dump/raw/test/uttblist_idx_entity"
    # blist_xphone_path = "./local/contextual/ssl_features/rareword_f10_test.xphone.seq.pt"
    blist_xphone_path = "./local/contextual/ssl_features/esun.entity.xphone.seq.pt"
    ref_path   = "./data/test/text"

    # AISHELL
    # scp_path   = "/home/ubuntu/espnet/egs2/aishell/asr1_contextual/dump/raw/test/wav.scp"
    # ref_path   = "/home/ubuntu/espnet/egs2/aishell/asr1_contextual/data/test/text"

    # rare_path  = "/home/ubuntu/espnet/egs2/aishell/asr1_contextual/local/contextual/rarewords/rareword_f10_test.txt"
    # blist_path = "/home/ubuntu/espnet/egs2/aishell/asr1_contextual/dump/raw/test/uttblist_idx"
    # blist_xphone_path = "/home/ubuntu/espnet/egs2/aishell/asr1_contextual/local/contextual/ssl_features/rareword_f10_test.xphone.seq.pt"
    

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
        'blist_xphone_path': blist_xphone_path,
        'blist_max': 100,
        'blist_drop_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
        # 'sampling_method': 'ann_hnw',
        # 'sampler_drop': 0.0,
        # 'hnwr_pre_gold_length': 5,
        # 'use_gpu': False,
        'use_oov': True,
        'prompt_template_context': '今天的主題為',
        'prompt_template_no_context': '好! 我們開始吧.', 
    }
    
    model, loader, contextual_processor = load_espnet_model(
        model_conf,
        contextual_conf, 
        token_path,
        'default', 
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type,
        use_local_attn_conv=False,
        return_contextual_processor=True,
    )
    preprocessor       = loader.dataset.preprocess
    tokenizer          = preprocessor.tokenizer
    token_id_converter = preprocessor.token_id_converter
    token_list         = get_token_list(token_id_converter) + ['<no-context>']

    model.contextualizer.retriever.temperature = 0.1
    if contextual_processor.sampling_method is not None:
        contextual_processor.asr_model.contextualizer = model.contextualizer
        contextual_processor.hn_sampler.update_index()

    print(model)

    model.eval()
    count   = 0
    results = {}
    for data in loader:
        if count >= 20:
            break
        count += 1

        uid = data[0][0]
        # if uid != "esun2022Q2_17":
        #     continue
        
        data = data[1]
        contexts          = data['contexts']
        speech            = data['speech']
        speech_lengths    = data['speech_lengths']
        text              = texts[uid]
        blist             = contexts['blist']
        ilens             = contexts['ilens']
        label_ctc         = contexts['label_ctc']

        print(f'texts: {text}')
        print(f'uid: {uid}')
        print(f'label_ctc:\n{label_ctc}')

        _blist = []
        for rareword in blist:
            btokens = [token_list[word] for word in rareword if word != -1]
            btokens = tokenizer.tokens2text(btokens)
            _blist.append(btokens)
        blist = _blist

        tokens = torch.tensor(
            preprocessor._text_process(
                {'text': text}
        )['text']).long()

        tokens = tokens.unsqueeze(0)

        logp, target, attens, ctc_pred, predict = forward(
            model, 
            speech, 
            speech_lengths,
            contexts,
            tokens,
            text,
            token_list,
        )
        results[uid] = predict
        
        for atten, tag in zip(attens, ['combine', 'sw', 'pho']):
            visualize(
                logp,
                atten,
                ctc_pred,
                text,
                target,
                blist,
                speech,
                model.blank_id, 
                token_list,
                debug_path,
                f'{uid}_{tag}',
            )
            # break
    debug_out_path = os.path.join(debug_path, 'predict')
    if not os.path.isdir(debug_out_path):
        os.makedirs(debug_out_path)
    
    output_path = os.path.join(debug_out_path, 'result.json')
    write_json(output_path, results)