import os
import jieba

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import write_file

ENTITY_LIST_PATH = f"./local/contextual/rarewords/esun.entity.sep.txt"

def is_phrase_in_sentence(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            return True
    return False

def get_uttblist(words, entity_phrases):
    # return [[str(word2idx[word]), word] for word in words if word in blist]

    # Segment the sentence using jieba
    sentence               = "".join(words)
    segmented_sentence     = list(jieba.cut(sentence))
    segmented_sentence_str = " ".join(segmented_sentence)

    # Set to keep track of detected phrases
    detected_phrases = []

    # Check if each phrase is present in the segmented sentence
    for phrase in entity_phrases:
        segmented_phrase = list(jieba.cut("".join(phrase.split(' '))))
        segmented_phrase_str = " ".join(segmented_phrase)
        if is_phrase_in_sentence(segmented_phrase, segmented_sentence):
            if not any(segmented_phrase_str in detected for _, detected in detected_phrases):
                detected_phrases.append([str(word2idx[phrase]), phrase])
    return detected_phrases

if __name__ == '__main__':
    datas_path = './dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        blist_path = ENTITY_LIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=',')]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...')
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        rareword_idxs  = []
        for data in tqdm(text_datas):
            uttid    = data[0]
            results  = get_uttblist(data[1:], blist)
            uttblist     = [d[1] for d in results]
            uttblist_idx = [d[0] for d in results]
            rareword_datas.append(
                [uttid] + (uttblist if len(uttblist) > 0 else [''])
            )
            rareword_idxs.append(
                [uttid] + (uttblist_idx if len(uttblist_idx) > 0 else [''])
            )

        output_path_uttblist = os.path.join(path, f'uttblist_entity')
        write_file(output_path_uttblist, rareword_datas)

        output_path_uttblist_idx = os.path.join(path, f'uttblist_idx_entity')
        write_file(output_path_uttblist_idx, rareword_idxs)