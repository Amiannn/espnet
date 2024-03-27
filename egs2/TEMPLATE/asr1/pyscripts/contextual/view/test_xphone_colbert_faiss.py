import os
import faiss
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

    # rare_path         = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/local/contextual/rarewords/rareword_f15.txt"
    # blist_xphone_path = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_contextual/local/contextual/ssl_features/rareword_f15.xphone.seq.pt"
    rare_path         = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/rarewords/rareword.train.sep.txt"
    blist_xphone_path = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/ssl_features/rareword.train.sep.xphone.seq.pt"

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
    
    blist_words    = [bword.replace(' ', '').lower() for bword in contextual_processor.blist_words]
    blist_size     = len(blist_words)
    xphones        = contextual_processor.blist_xphone
    xphone_indexis = contextual_processor.blist_xphone_indexis
    
    flatten_idx = []
    for i, idx in enumerate(xphone_indexis):
        start, end = idx
        flatten_idx.extend([i] * (end - start))
    flatten_idx = torch.tensor(flatten_idx)

    print(f'blist_xphone shape: {xphones.shape}')
    print(f'flatten_idx shape : {flatten_idx.shape}')

    # build index
    s, d  = xphones.shape
    index = faiss.IndexFlat(d)
    index.add(xphones)

    # gpu
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # search
    datas = []
    for i in tqdm(range(blist_size)):
        start, end = xphone_indexis[i]
        xphone     = xphones[start:end, :]
        
        topk = 5
        D, I = index.search(xphone, topk + 1)
        I    = flatten_idx[I[:, 1:]]

        uni_idx, counts = torch.unique(I, return_counts=True)
        count_idx       = torch.argsort(counts).tolist()[::-1]

        nbest = []
        for k in count_idx:
            idx = uni_idx[k]
            nbest.append(blist_words[idx])

        result = f'(colbert search)\t{blist_words[i]}: {", ".join(nbest)}\n'
        datas.append([result])

    output_path = './xphone.colbert.seq'
    write_file(output_path, datas, sp=' ')

