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

from sklearn.metrics import precision_recall_fscore_support

input_path = './exp/asr_whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch/debug/7epoch/predicts_ami.json'
# input_path = './exp/asr_whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch/debug/7epoch/predicts_200.json'
result = read_json(input_path)

def precision_recall(y_true, y_pred):
    y_true_length = len(y_true)
    y_pred_length = len(y_pred)
    counts    = sum([1 if p in y_true else 0 for p in y_pred])
    precision = (counts / y_pred_length) if y_pred_length != 0 else 1
    counts    = sum([1 if t in y_pred else 0 for t in y_true])
    recall    = (counts / y_true_length) if y_true_length != 0 else 1
    return recall, precision

recalls, precisions = [], []
for uid in result:
    y_true = result[uid]['label']
    y_pred = result[uid]['pred']
    
    # if len(y_true) == 0 and len(y_pred) == 0:
    #     continue
    recall, precision = precision_recall(y_true, y_pred)
    recalls.append(recall)
    precisions.append(precision)

recalls    = np.mean(recalls)
precisions = np.mean(precisions)
f1 = 2 * precisions * recalls / (precisions + recalls)
print(f'recall: {recalls}')
print(f'precision: {precisions}')
print(f'f1: {f1}')