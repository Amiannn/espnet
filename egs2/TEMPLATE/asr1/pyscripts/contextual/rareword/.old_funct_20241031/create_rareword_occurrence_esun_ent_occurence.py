import os
import jieba
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/rarewords/esun.entity.sep.txt"

def is_phrase_in_sentence(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            return True
    return False

def get_uttblist(segmented_sentence, entity_phrase):
    # Check if each phrase is present in the segmented sentence
    segmented_phrase = list(jieba.cut("".join(entity_phrase.split(' '))))
    segmented_phrase_str = " ".join(segmented_phrase)
    if is_phrase_in_sentence(segmented_phrase, segmented_sentence):
        return True
    return False

def occurrence(texts, bwords):
    # Segment the sentence using jieba
    bword_occurrence = {word: 0 for word in bwords}
    # for _, words in tqdm(texts):
    #     sentence           = "".join(words)
    #     segmented_sentence = list(jieba.cut(sentence))
    #     for bword in bwords:
    #         check = get_uttblist(segmented_sentence, bword)
    #         if check:
    #             print(f'hit! {bword}')
    #             bword_occurrence[bword] += 1
    print(f'length: {len(list(bword_occurrence.values()))}')
    return list(bword_occurrence.values())

if __name__ == '__main__':
    text_path  = './dump/raw/train_sp/text'
    dev_path   = './dump/raw/dev/text'
    dump_path  = './local/contextual/rarewords'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')

    text_datas = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    dev_datas  = [[d[0], d[1:]] for d in read_file(dev_path, sp=' ')]
    blist      = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=',')]
    print(f'blist: {len(blist)}')
    # text_datas = text_datas + dev_datas
    text_datas = text_datas
    counts     = list(map(lambda x: [str(x)], occurrence(text_datas, blist)))

    output_path = os.path.join(dump_path, f'{name}_occurrence_.txt')
    write_file(output_path, counts)

    bword_occurrence = {word: 0 for word in blist}
    bwords = [[k] for k in bword_occurrence.keys()]
    output_path = os.path.join(dump_path, f'{name}_.txt')
    write_file(output_path, bwords)
