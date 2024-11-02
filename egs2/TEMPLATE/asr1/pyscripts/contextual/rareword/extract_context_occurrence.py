import os
import jieba
import numpy as np

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TRAIN_DEV_BLIST_PATH = f"./local/contextual/rarewords/esun_earningcall.entity.txt"

def is_phrase_in_sentence(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            return True
    return False

def get_uttblist(segmented_sentence, entity_phrase):
    # Check if each phrase is present in the segmented sentence
    segmented_phrase = list(jieba.cut(entity_phrase.replace(' ', '')))
    if is_phrase_in_sentence(segmented_phrase, segmented_sentence):
        return True
    return False

def occurrence(texts, bwords):
    # Segment the sentence using jieba
    bword_occurrence = {word: 0 for word in bwords}
    for _, words in tqdm(texts):
        sentence           = "".join(words).lower()
        segmented_sentence = list(jieba.cut(sentence))
        for bword in bwords:
            check = get_uttblist(segmented_sentence, bword)
            if check:
                bword_occurrence[bword] += 1
    result = [0 for _ in bwords]
    for bword in bword_occurrence:
        index = word2idx[bword]
        result[index] = bword_occurrence[bword]
    return result

if __name__ == '__main__':
    text_path  = './dump/raw/train_sp/text'
    test_path  = './dump/raw/test/text'
    dump_path  = './local/contextual/rarewords'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')
    
    jieba.load_userdict(TRAIN_DEV_BLIST_PATH)

    text_datas_train = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    text_datas_test  = [[d[0], d[1:]] for d in read_file(test_path, sp=' ')]
    blist    = [b[0].lower() for b in read_file(TRAIN_DEV_BLIST_PATH, sp=',')]
    word2idx = {bword: i for i, bword in enumerate(blist)}
    blist    = sorted(blist, key=lambda b: len(b), reverse=True)

    counts = list(map(lambda x: [str(x)], occurrence(text_datas_train, blist)))
    output_path = os.path.join(dump_path, f'{name}_occurrence_train.txt')
    write_file(output_path, counts)

    counts = list(map(lambda x: [str(x)], occurrence(text_datas_test, blist)))
    output_path = os.path.join(dump_path, f'{name}_occurrence_test.txt')
    write_file(output_path, counts)