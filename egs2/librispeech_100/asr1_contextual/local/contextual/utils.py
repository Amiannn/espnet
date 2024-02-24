import os
import json
import time
import pickle

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

def write_file(path, datas, sp=" "):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join(data) + '\n')

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return json.load(fr)

def write_json(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        json.dump(datas, fr, indent=4, ensure_ascii=False)

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def write_pickle(path, datas):
    with open(path, 'wb') as f:
        pickle.dump(datas, f)