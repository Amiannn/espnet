import os
import json
import numpy as np
import matplotlib.pyplot as plt

from jiwer import cer
from tqdm  import tqdm

from rapidfuzz    import fuzz
from pypinyin     import lazy_pinyin
from numpy.linalg import norm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from pyscripts.contextual.utils.aligner import CheatDetector
from pyscripts.contextual.utils.aligner import align_to_index

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

def similarity(query, value):
    score = fuzz.ratio(query, value) / 100
    return score

def cos_similarity(query, value):
    return 1 - ((np.dot(query, value)/(norm(query) * norm(value)) + 1) / 2)

if __name__ == '__main__':
    entity_path             = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/rarewords/rareword.all.sep.txt"
    entity_phone_path       = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/local/contextual/ssl_features/rareword.all.sep.pho.txt"
    entity_description_path = "/share/nas165/amian/experiments/speech/Dancer_dev/datas/entities/aishell/descriptions/ctx.json"

    entity_datas     = [d[0].replace(' ', '') for d in read_file(entity_path, sp='\t')]
    entity_pho_datas = [d[0].replace(' ', '') for d in read_file(entity_phone_path, sp='\t')]
    entity_des_datas = [[d['entity'], d['intro']] for d in read_json(entity_description_path)]
    ent2idx          = {ent: i for i, ent in enumerate(entity_datas)}

    entities = []
    for ent, des in entity_des_datas:
        idx = ent2idx[ent]
        entities.append([ent, entity_pho_datas[idx], des])

    print(entities[:3])
    print(len(entities))

    entity_size = len(entities)
    # pho_sim_dict = {} 
    # for i in tqdm(range(entity_size)):
    #     source_ent, source_pho, source_des = entities[i]
    #     for j in range(i + 1, entity_size, 1):
    #         target_ent, target_pho, target_des = entities[j]
    #         score = similarity(source_pho, target_pho)
    #         pho_sim_dict[f'{i}_{j}'] = score
    # print(f'pho_sim_dict: {pho_sim_dict}')

    # output_path = './aishell_pho_sim.pkl'
    # write_pickle(output_path, pho_sim_dict)

    # model      = SentenceTransformer('sentence-transformers/LaBSE')
    # embeddings = []
    # for ent, pho, des in tqdm(entities):
    #     embedding = model.encode([des])
    #     embeddings.append(embedding)
    # embeddings = np.stack(embeddings).reshape(-1, 768)
    # output_path = './aishell_sem_vec.pkl'
    # write_pickle(output_path, embeddings)
    # print(f'embeddings: {embeddings.shape}')
    
    # embeddings = read_pickle('./aishell_sem_vec.pkl')
    # print(f'embeddings: {embeddings.shape}')

    # sem_sim_dict = {}
    # for i in tqdm(range(entity_size)):
    #     source_embed = embeddings[i]
    #     for j in range(i + 1, entity_size, 1):
    #         target_embed = embeddings[j]
    #         score = cos_similarity(source_embed, target_embed)
    #         sem_sim_dict[f'{i}_{j}'] = score
    # print(f'sem_sim_dict: {sem_sim_dict}')

    # output_path = './aishell_sem_sim.pkl'
    # write_pickle(output_path, sem_sim_dict)

    # print('loading pho dict...')
    # pho_sim_dict = read_pickle('./aishell_pho_sim.pkl')
    # print('loading sem dict...')
    # sem_sim_dict = read_pickle('./aishell_sem_sim.pkl')
    # print(f'pho_sim_dict: {pho_sim_dict}')
    # print(f'sem_sim_dict: {sem_sim_dict}')

    # scores = []
    # for key in tqdm(pho_sim_dict):
    #     pho_score = pho_sim_dict[key]
    #     sem_score = sem_sim_dict[key]
    #     scores.append([pho_score, sem_score])

    # scores = sorted(scores, key=lambda s:s[0])
    # scores = [[str(p), str(s)] for p, s in scores]
    # output_path = './aishell_sem_pho_confusion.csv'
    # write_file(output_path, scores, sp=',')

    X, Y   = [], []
    xx, yy = [], []
    count = 0
    with open('./aishell_sem_pho_confusion.csv', 'r', encoding='utf-8') as frs:
        for fr in tqdm(frs):
            if (count + 1) % 1000 == 0:
                X.append(np.mean(xx))
                Y.append(np.mean(yy))
                xx, yy = [], []
            # if count > 100000000:
            #     break
            pho_score, sem_score = map(float, (fr.replace('\n', '')).split(','))
            xx.append(pho_score)
            yy.append((1 - sem_score))
            count += 1
    X.append(np.mean(xx))
    Y.append(np.mean(yy))

    plt.plot(X, Y)
    output_path = os.path.join('aishell_pho_sem_plot.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
