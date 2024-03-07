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
from pyscripts.contextual.utils.rnnt_alignment import forward_backward as alignment

from espnet2.asr_transducer.utils import get_transducer_task_io

seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

@torch.no_grad()
def forward(model, contextual_processor, speech, text):
    speech  = speech.unsqueeze(0)
    lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    encoder_out, enc_olens = model.encode(speech, lengths)
    print(f'encoder_out: {encoder_out.shape}')

    decoder_in, target, t_len, u_len = get_transducer_task_io(
        tokens,
        enc_olens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )
    # decoder_out = model.decoder(decoder_in)
    # print(f'decoder_out: {decoder_out.shape}')

if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600/tokens.txt"
    model_conf = "./conf/exp/contextual_adapter/train_rnnt_contextual_adapter_encoder.yaml"
    model_path = "./exp/asr_train_rnnt_contextual_adapter_encoder_raw_en_bpe600_use_wandbtrue_wandb_projectContextualize_RNNT_sp_suffix/valid.loss.ave_10best.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_sp_suffix/train/feats_lengths_stats.npz"
    rare_path  = "./local/contextual/rareword_f15.txt"
    scp_path   = "./dump/raw/test_clean/wav.scp"
    blit_path  = "./dump/raw/test_clean/uttblist"
    ref_path   = "./data/test_clean/text"

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    data_path_and_name_and_type = [
        (scp_path, 'speech', 'kaldi_ark'), 
        (blit_path, 'uttblist', 'multi_columns_text')
    ]

    contextual_conf = {
        'contextual_type': 'rareword',
        'blist_path': './local/contextual/rareword_f15.txt',
        'blist_max': 5,
        'blist_droup_out': 0.0,
        'warmup_epoch': 0,
        'structure_type': None,
        'sampling_method': None,
    }
    
    (
        model, 
        bpemodel, 
        tokenizer, 
        converter, 
        loader
    ) = load_espnet_model(
        model_conf,
        contextual_conf, 
        token_path, 
        stats_path, 
        spm_path, 
        model_path,
        data_path_and_name_and_type
    )

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
        break