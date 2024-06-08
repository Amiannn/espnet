import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file
from pyscripts.contextual.utils.dataio import write_json

def build(dataset_names):
    data = {}
    for dataset_name in dataset_names:
        for idx in dataset2idx[dataset_name]:
            entity_dict = utterance_entity_datas[idx]
            for ent in entity_dict:
                desc = f"{ent}: {entity_dict[ent]['desc']}".lower()
                data[desc] = ent
    data = sorted([[data[ent], ent] for ent in data])
    return data

root_folder  = "/share/nas165/amian/experiments/speech/espnet/egs2/earningcall/asr1_contextual"
output_dir   = "./local/contextual/metadatas"
dataset_path = "data"
utterance_entity_path = [
    "local/contextual/metadatas/earning21_utterances.json",
    "local/contextual/metadatas/earning22_utterances.json",
]
metadata_path = [
    "local/contextual/metadatas/earning21_metadata_final.json",
    "local/contextual/metadatas/earning22_metadata_final.json"
]

utterance_entity_datas = []
for meta_path in metadata_path:
    path = os.path.join(root_folder, meta_path)
    data = read_json(path)
    utterance_entity_datas.extend(data)
utterance_entity_datas = {data['id']: data['entity'] for data in utterance_entity_datas}

utterance_datas = {}
for uttent_path in utterance_entity_path:
    path = os.path.join(root_folder, uttent_path)
    data = read_json(path)
    utterance_datas.update(data)

print(f'utterance_entity_datas: {len(utterance_entity_datas)}')
print(f'utterance_datas: {len(utterance_datas)}')

dataset_names = {
    'train': 'train',
    'dev'  : 'validation',
    'test' : 'test'
}

dataset2idx = {}
for dataset_name in dataset_names:
    path  = os.path.join(root_folder, dataset_path, dataset_name, 'text')
    ids   = [d[0].split('-')[-1].split('_')[0] for d in read_file(path, ' ')]
    dataset2idx[dataset_name] = list(set(ids))

dataset = build({'train': 'train'})
output_path = os.path.join(output_dir, 'entity_description_train.json')
write_json(output_path, dataset)
entity_datas = [[d[0]] for d in dataset]
output_path = os.path.join('./local/contextual/rarewords', 'entity_train.text')
write_file(output_path, entity_datas)

dataset = build({'dev': 'validation'})
output_path = os.path.join(output_dir, 'entity_description_dev.json')
write_json(output_path, dataset)
entity_datas = [[d[0]] for d in dataset]
output_path = os.path.join('./local/contextual/rarewords', 'entity_dev.text')
write_file(output_path, entity_datas)

dataset = build({'test': 'test'})
output_path = os.path.join(output_dir, 'entity_description_test.json')
write_json(output_path, dataset)
entity_datas = [[d[0]] for d in dataset]
output_path = os.path.join('./local/contextual/rarewords', 'entity_test.text')
write_file(output_path, entity_datas)

dataset = build({'train': 'train', 'dev': 'validation'})
output_path = os.path.join(output_dir, 'entity_description_train_dev.json')
write_json(output_path, dataset)
entity_datas = [[d[0]] for d in dataset]
output_path = os.path.join('./local/contextual/rarewords', 'entity_train_dev.text')
write_file(output_path, entity_datas)

dataset = build({'train': 'train', 'dev': 'validation', 'test': 'test'})
output_path = os.path.join(output_dir, 'entity_description_all.json')
write_json(output_path, dataset)
entity_datas = [[d[0]] for d in dataset]
output_path = os.path.join('./local/contextual/rarewords', 'entity_all.text')
write_file(output_path, entity_datas)