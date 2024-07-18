import os
import torch
import random
import torchaudio
import numpy as np

from tqdm import tqdm
from fuzzywuzzy import fuzz

from sklearn.metrics.pairwise import cosine_similarity

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-uncased')
cache     = {}

def encode(word):
    if word in cache:
        return cache[word]
    tokens = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    vec = outputs.last_hidden_state[:, 0, :].numpy()
    cache[word] = vec
    return vec

def bert_similarity(word1, word2):
    vec1 = encode(word1)
    vec2 = encode(word2)
    return cosine_similarity(vec1, vec2)[0][0]

pattern_path = './exp/analysis/error_patterns_pho.json'
dump_path    = './exp/analysis'

if __name__ == '__main__': 
    datas    = read_json(pattern_path)
    patterns = datas['errors']

    scores_list = []
    for word in tqdm(list(patterns.keys())[:100]):
        errors = patterns[word]['errors']
        score_list     = []
        for hyp, hyp_pho in errors:
            score     = bert_similarity(word, hyp)
            score_list.append(score)
            print(f'ref: {word}, hyp: {hyp}, score: {score}')
        scores_list.append(
            sum(score_list) / len(score_list)
        )
        print(f'_' * 30)
    scores = sum(scores_list) / len(scores_list)
    print(f'scores : {scores}')