import os
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file
from pyscripts.contextual.utils.dataio import write_json

BLIST_IDX_PATH = "./dump/raw/train_sp/uttblist_idx"
TEXT_PATH      = "./dump/raw/train_sp/text"
ENTITY_PATH    = "./local/contextual/rarewords/entity.train.sep.txt"

uttblist_datas = {d[0]: [int(i) for i in d[1:] if i != ""] for d in read_file(BLIST_IDX_PATH, sp=' ')}
text_datas     = {d[0]: d[1] for d in read_file(TEXT_PATH, sp=' ')}
entity_datas   = [d[0] for d in read_file(ENTITY_PATH, sp='\t')]

segments    = {}
segment_ids = {}
for uid in uttblist_datas:
    if uid.startswith('sp'):
        continue
    idxs    = uttblist_datas[uid]
    utt_ent = [entity_datas[idx] for idx in idxs]
    _uid = uid.split('W')[0]
    # segments[uid] = segments[uid] + [utt_ent] if uid in segments else [utt_ent]
    text = f'{text_datas[uid]}'
    segments[_uid] = segments[_uid] + text if _uid in segments else text
    text = f'{uid}: {text_datas[uid]}\n'
    segment_ids[_uid] = segment_ids[_uid] + text if _uid in segment_ids else text

# print(segments)
print(f'segments length: {len(segments)}')

output_path = './exp/test/longform_train.json'
write_json(output_path, segments)

# from googlesearch import search

# query = "这些高净值人士都是上述房企海外项目的潜在客群多家在海外开发项目的房企"
# num_results = 3

# results = []
# for sid in segments:
#     text = segments[sid]
#     chunk_div = len(text) // 200
#     chunks, chunk_size = len(text), len(text) // chunk_div

#     sub_texts = [text[i:i+chunk_size] for i in range(0, chunks, chunk_size)]
#     print(f'chunk_size: {chunk_size}')
#     for query in sub_texts:
#         results = [result for result in search(
#             query, 
#             stop=num_results, 
#             pause=1.0,
#             lang="zh-CN", 
#         )]
#         print(results)

#         output_path = './exp/test/test_gs.json'
#         write_json(output_path, results)

from openai import OpenAI

template = '''
Question: 請幫我將以下的內容依照主題分段落

Output format (section start id, section end id):
{
    1: ["BAC009S0002", "BAC009S0090"],
    2: ["BAC009S0091", "BAC009S0190"],
    3: ["BAC009S0190", "BAC009S0300"],
}

Document:
'''

client = OpenAI()
print(segment_ids['BAC009S0002'])
response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": template+segment_ids['BAC009S0002']+"\n Output:\n"}
  ]
)
print(response.choices[0].message.content)