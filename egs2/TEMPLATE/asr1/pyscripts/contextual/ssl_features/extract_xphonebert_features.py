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
rareword_path = "./local/contextual/rarewords/all_rare_words.txt"
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

input_phonemes_list = []
print(f'converting word to phoneme sequence...')
for sentence in tqdm(rarewords):
    input_phonemes = text2phone_model.infer_sentence(sentence)
    input_phonemes_list.append(input_phonemes)

print(f"input_phonemes_list: {input_phonemes_list}")
output_pho_path = os.path.join(output_path, f'{filename}.pho.txt')
input_phonemes_data = [[pho] for pho in input_phonemes_list]
write_file(output_pho_path, input_phonemes_data)

features_list = []
print(f'encoding phonemes...')
for input_phonemes in tqdm(input_phonemes_list):
    input_ids = tokenizer(input_phonemes, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        features = xphonebert(**input_ids)
    features = features.last_hidden_state[:, 1:-1, :].to('cpu')
output_path = os.path.join(output_path, f'{filename}.xphone.pt')
torch.save(features, output_path)