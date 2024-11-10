import os
import numpy as np

from tqdm import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

def read_entity(path):
    datas = read_file(path, ' ')
    result = []

    for data in datas:
        data = [d for d in data[1:] if d != ""]
        if len(data) > 0:
            result.extend(data)
    return list(set(result))

train_entity_path = "./dump/raw/train_sp/uttblist_idx_entity"
dev_entity_path   = "./dump/raw/dev/uttblist_idx_entity"
test_entity_path  = "./dump/raw/test/uttblist_idx_entity"

train_datas = read_entity(train_entity_path)
dev_datas   = read_entity(dev_entity_path)
test_datas  = read_entity(test_entity_path)

print(f'train_datas length: {len(train_datas)}')
print(f'dev_datas length  : {len(dev_datas)}')
print(f'test_datas length : {len(test_datas)}')

train_dev_datas = list(set(train_datas + dev_datas))

zero_shot_datas = list(set(test_datas) - set(train_dev_datas))
print(f'zero_shot_datas length : {len(zero_shot_datas)}')

entity_path = "./local/contextual/rarewords/esun.entity.sep.txt"

entity_datas = [d[0].replace(" ", "") for d in read_file(entity_path)]

zero_shot_entity = [entity_datas[int(i)] for i in zero_shot_datas]
print(f'zero_shot_entity: {zero_shot_entity}')

entity_description_path = "./local/contextual/metadatas/esun.entity.json"
entity_desc_datas = {d['entity']: d for d in read_json(entity_description_path)}

datas = [entity_desc_datas[ent] for ent in zero_shot_entity]
output_path = "./local/contextual/metadatas/esun.entity.zeroshot.json"
write_json(output_path, datas)