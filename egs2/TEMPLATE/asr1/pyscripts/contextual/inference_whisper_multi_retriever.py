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
        pred_ctc  = [token_list[p] for p in ctc_pred]
        _pred_ctc = [token_list[p] for p in ctc_pred if p != 0]
        _texts  = list(tokenizer.tokens2text(_pred_ctc))
        print(f'_texts: {_texts}')
        texts  = list(tokenizer.tokens2text(pred_ctc))
        _texts = ['.' for _ in range(max([len(pred_ctc), len(texts)]))]
        last   = '!'
        for i, t in enumerate(texts):
            if t == last or t == '!':
                _texts[i] = ' '
            else:
                _texts[i] = t
            last = t
        frame2align = {i: _texts[i] for i in range(atten.shape[1])}
        print(f'pred_ctc: {len(pred_ctc)}, texts: {len(texts)}')
        # print(f'pred_ctc: {pred_ctc}')

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
        print(f'context_prob: {context_prob.shape}')
        predict = topk_decode(
            context_prob, 
            blist, 
            idx_blank=0, 
            top_k=5, 
            threshold=0.6
        )
        predict_ctc = retrieve_ctc_decode(
            context_prob,
            blist,
            idx_blank=0,
        )
        print(f'Retriever predict: {predict}')
    predict = {
        'text'      : text,
        'result'    : predict,
        'result_ctc': predict_ctc,
    }
    ctc_pred = None
    if model.ctc is not None:
        print(f'encoder_proj: {encoder_proj.shape}')
        x        = encoder_proj if encoder_proj is not None else encoder_out
        ctc_pred = model.ctc.argmax(x).squeeze(0)
        print(f'ctc_pred: {ctc_pred.shape}')

    logp        = None
    target      = None
    return logp, target, context_prob, ctc_pred, predict

if __name__ == "__main__":
    spm_path   = "whisper_multilingual"
    token_path = "./data/zh_token_list/whisper_multilingual/tokens.txt"
    model_conf = "./conf/contextual/whisper/train_asr_whisper_medium_conv2_multilateinteractive_retrieval_balanced_alpha0.5_annhnw_biglist.yaml"
    model_path = "./exp/asr_whisper/run_medium_conv2_multilateinteractive_retrieval_balanced_alpha0.5_annhnw_biglist/2epoch.pth"
    # model_path = "./exp/asr_whisper/run_medium_multilateinteractive_retrieval_balanced_alpha0.8_annhnw/valid.loss.ave_10best.pth"
    stats_path = None

    # rare_path  = "./local/contextual/rarewords/rareword_f10_test.txt"
    rare_path  = "./local/contextual/rarewords/esun.entity.txt"
    scp_path   = "./dump/raw/test/wav.scp"
    # blist_path = "./dump/raw/test/uttblist_idx"
    blist_path = "./dump/raw/test/uttblist_idx_entity"
    ref_path   = "./data/test/text"

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
        # 'blist_xphone_path': './local/contextual/ssl_features/rareword_f10_test.xphone.seq.pt',
        'blist_xphone_path': './local/contextual/ssl_features/esun.entity.xphone.seq.pt',
        'blist_max': 200,
        'blist_drop_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
        # 'sampling_method': 'ann_hnw',
        # 'sampler_drop': 0.0,
        # 'hnwr_pre_gold_length': 5,
        # 'use_gpu': False,
        # 'use_oov': True,
        'prompt_template_context': '今天的主題為',
        'prompt_template_no_context': '好! 我們開始吧.', 
    }
    
    model, loader, contextual_processor = load_espnet_model(
        model_conf,
        contextual_conf, 
        token_path,
        None, 
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type,
        use_local_attn_conv=False,
        token_type='whisper_multilingual',
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
        nlp_prompt_tensor = contexts['nlp_prompt_tensor']

        print(f'texts: {text}')
        print(f'uid: {uid}')
        print(f'label_ctc:\n{label_ctc}')

        nlp_prompt_tokens = [token_list[p] for p in nlp_prompt_tensor]

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

        logp, target, atten, ctc_pred, predict = forward(
            model, 
            speech, 
            speech_lengths,
            contexts,
            tokens,
            text
        )
        results[uid] = predict
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
            uid,
        )
        # break
    debug_out_path = os.path.join(debug_path, 'predict')
    if not os.path.isdir(debug_out_path):
        os.makedirs(debug_out_path)
    
    output_path = os.path.join(debug_out_path, 'result.json')
    write_json(output_path, results)