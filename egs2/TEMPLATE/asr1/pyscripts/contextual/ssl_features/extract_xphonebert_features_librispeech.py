# https://github.com/VinAIResearch/XPhoneBERT

import os
import torch
import numpy as np

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_pickle
from pyscripts.contextual.utils.dataio import write_json
from pyscripts.contextual.utils.dataio import write_file
from pyscripts.contextual.utils.dataio import write_pickle

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

model_tag     = "vinai/xphonebert-base"
rareword_path = "./local/contextual/rarewords/rareword_f15.txt"
output_path   = "./local/contextual/ssl_features"

filename  = (rareword_path.split('/')[-1]).split('.')[0]
rarewords = [d[0].lower() for d in read_file(rareword_path, sp=' ')]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained(model_tag)
xphonebert = xphonebert.to(device)
tokenizer  = AutoTokenizer.from_pretrained(model_tag)

# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)

print(f'converting word to phoneme sequence...')
pho_path = os.path.join(output_path, f'{filename}.pho.txt')
if os.path.isfile(pho_path):
    input_phonemes_list = [d[0] for d in read_file(pho_path, sp=',')]
else:
    input_phonemes_list = []
    for sentence in tqdm(rarewords):
        input_phonemes = text2phone_model.infer_sentence(sentence)
        input_phonemes_list.append(input_phonemes)

    output_pho_path = os.path.join(output_path, f'{filename}.pho.txt')
    input_phonemes_data = [[pho] for pho in input_phonemes_list]
    write_file(output_pho_path, input_phonemes_data)

print(f"input_phonemes_list: {input_phonemes_list[:10]}")

features_list = []
features_lens = []
now_idx       = 0
print(f'encoding phonemes...')
for input_phonemes in tqdm(input_phonemes_list):
    input_ids = tokenizer(input_phonemes, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        features = xphonebert(**input_ids)
    features = features.last_hidden_state.squeeze(0).to('cpu')
    ilens, D = features.shape
    features_list.append(features)
    features_lens.append([now_idx, now_idx + ilens])
    now_idx += ilens
features_list = torch.cat(features_list, dim=0)
features_lens = torch.tensor(features_lens).long()
print(f'features list shape: {features_list.shape}')
print(f'features ilen shape: {features_lens.shape}')
output_path = os.path.join(output_path, f'{filename}.xphone.seq.pt')
torch.save(
    {
        'features': features_list,
        'indexis' :  features_lens
    }, 
    output_path
)