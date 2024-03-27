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
from torch.nn.utils.rnn import pad_sequence

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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
        contextualizer      = model.contextualizer
        context_idxs        = contexts['blist']
        context_xphone_idxs = contexts['blist_xphone']
        ilens               = contexts['ilens']

        print(f'context_xphone_idxs: {context_xphone_idxs.shape}')

    # decoder_in, target, t_len, u_len = get_transducer_task_io(
    #     tokens,
    #     enc_olens,
    #     ignore_id=-1,
    #     blank_id=model.blank_id,
    # )
    # decoder_out = model.decoder(decoder_in)

    # joint_out = model.joint_network(
    #     encoder_out.unsqueeze(2), 
    #     decoder_out.unsqueeze(1),
    #     bias_out=bias_vec.unsqueeze(2),
    # )
    # logp = torch.log_softmax(joint_out, dim=-1)[0].transpose(1, 0)

    # return logp, target, atten

def scoring(score, blist_words, k=15):
    indexis = torch.argsort(score).tolist()[::-1][:k]
    nbest   = [blist_words[index].replace(' ', '') for index in indexis]
    return nbest

if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600/tokens.txt"
    model_conf = "./conf/exp/contextual_adapter/train_rnnt_contextual_adapter_tf_xphone_encoder_with_gactc.yaml"
    model_path = "./exp/asr_finetune_freeze_ct_enc_cb_tf_xphone_gactc_suffix/1epoch.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_sp_suffix/train/feats_lengths_stats.npz"
    scp_path   = "./dump/raw/test_clean/wav.scp"
    blist_path = "./dump/raw/test_clean/uttblist_idx"
    ref_path   = "./data/test_clean/text"

    rare_path         = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/local/contextual/rarewords/rareword_f15.txt"
    blist_xphone_path = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/local/contextual/ssl_features/rareword_f15.xphone.seq.pt"
    # rare_path         = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/rarewords/rareword.train.sep.txt"
    # blist_xphone_path = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/ssl_features/rareword.train.sep.xphone.seq.pt"
    
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
        'blist_xphone_path': blist_xphone_path,
        'blist_max': 20,
        'blist_droup_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
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
        return_contextual_processor=True
    )

    preprocessor       = loader.dataset.preprocess
    token_id_converter = preprocessor.token_id_converter
    token_list         = token_id_converter.token_list + ['<oov>']
    
    model.eval()
    blist_words          = [bword.lower() for bword in contextual_processor.blist_words]
    blist_xphone         = contextual_processor.blist_xphone
    blist_xphone_indexis = contextual_processor.blist_xphone_indexis
    blist_size           = len(blist_words)
    
    _blist_size = 1000

    blist_xphone_pad = pad_sequence(
        [blist_xphone[start:end, :] for start, end in blist_xphone_indexis], 
        batch_first=True, 
        padding_value=0
    )
    print(f'blist_xphone_pad shape: {blist_xphone_pad.shape}')

    blist_xphone_first = torch.stack(
        [blist_xphone[start, :] for start, end in blist_xphone_indexis]
    )
    print(f'blist_xphone_first shape: {blist_xphone_first.shape}')

    blist_xphone_mean = torch.stack(
        [torch.mean(blist_xphone[start+1:end-1, :], dim=0) for start, end in blist_xphone_indexis]
    )
    print(f'blist_xphone_mean shape: {blist_xphone_mean.shape}')

    nbest = []
    datas = []
    for idx in tqdm(range(_blist_size)):
        bword  = blist_words[idx].replace(' ', '')
        
        # full search
        xphone = blist_xphone_pad[idx]
        sim    = blist_xphone_pad @ xphone.T
        score  = torch.sum(torch.max(sim, dim=1).values, dim=-1)
        nbest  = scoring(score, blist_words, k=15)
        result = f'(Full search)\t{bword}: {", ".join(nbest)}\n'

        # <s> search
        xphone  = blist_xphone_first[idx]
        score   = blist_xphone_first @ xphone
        nbest   = scoring(score, blist_words, k=15)
        result += f'(<s> search)\t{bword}: {", ".join(nbest)}\n'

        # mean search
        xphone  = blist_xphone_mean[idx]
        score   = blist_xphone_mean @ xphone
        nbest   = scoring(score, blist_words, k=15)
        result += f'(Mean search)\t{bword}: {", ".join(nbest)}\n'

        datas.append([result])

    output_path = './xphone.en.seq'
    write_file(output_path, datas, sp=' ')