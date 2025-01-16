import os
import jieba

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyscripts.contextual.utils.dataio import read_file, write_file

filename = f'f{2 ** 16}'
test_filename = f'f{10}'

filename = "S95_keywords"
test_filename = "keywords"

TRAIN_DEV_BLIST_PATH = f"./local/contextual/contexts/context_{filename}_train.txt"
TEST_BLIST_PATH      = f"./local/contextual/contexts/context_{test_filename}_test.txt"

def init_worker(init_bl, init_w2i):
    """Initializer function for each worker in the pool."""
    global blist
    global word2idx
    blist = init_bl
    word2idx = init_w2i

def process_data(data):
    """Function to process each data entry."""
    uttid = data[0]
    text  = [d.lower() for d in data[1:]]
    # results = get_uttblist(text)
    results = get_uttblist_quick(text)
    uttblist = [d[1] for d in results]
    uttblist_idx = [d[0] for d in results]
    context_data = [uttid] + (uttblist if uttblist else [''])
    context_idx = [uttid] + (uttblist_idx if uttblist_idx else [''])
    return context_data, context_idx

def is_phrase_in_sentence(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            return True
    return False

def get_uttblist(words):
    # Segment the sentence using jieba
    sentence               = " ".join(words)
    segmented_sentence     = list(jieba.cut(sentence))

    # Set to keep track of detected phrases
    detected_phrases = []

    # Check if each phrase is present in the segmented sentence
    for phrase in blist:
        segmented_phrase = list(jieba.cut("".join(phrase.split(' '))))
        segmented_phrase_str = " ".join(segmented_phrase)
        if is_phrase_in_sentence(segmented_phrase, segmented_sentence):
            if not any(segmented_phrase_str in detected for _, detected in detected_phrases):
                detected_phrases.append([str(word2idx[phrase]), phrase])
        detected_phrases = sorted(detected_phrases, key=lambda d: sentence.find(d[-1]))
    return detected_phrases

# English only
def get_uttblist_quick(words):
    detected_phrases = []
    # Check if each phrase is present in the segmented sentence
    for word in words:
        if word in word2idx:
            detected_phrases.append([str(word2idx[word]), word])
    return detected_phrases

if __name__ == '__main__':
    datas_path = './dump/raw'
    # Set the number of worker processes
    num_workers = cpu_count()  # Use all available CPU cores
    # num_workers = 4          # Or set to a specific number
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')) or "L95_sp" in path:
            continue
        if 'test' in path:
            blist_path = TEST_BLIST_PATH
            now_filename = test_filename
        else:
            blist_path = TRAIN_DEV_BLIST_PATH
            now_filename = filename
        blist = [b[0].lower() for b in read_file(blist_path, sp=' ')]
        word2idx = {word: i for i, word in enumerate(blist)}
        # Sort by length
        blist = sorted(blist, key=lambda s: len(s), reverse=True)

        print(f'Processing {path}...')
        text_path = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')

        with Pool(processes=num_workers, initializer=init_worker, initargs=(blist, word2idx)) as pool:
            results = list(tqdm(pool.imap(process_data, text_datas), total=len(text_datas)))

        context_datas, context_idxs = zip(*results)
        context_datas = list(context_datas)
        context_idxs = list(context_idxs)

        output_path_uttblist = os.path.join(path, f'uttblist_{now_filename}')
        write_file(output_path_uttblist, context_datas)

        output_path_uttblist_idx = os.path.join(path, f'uttblist_idx_{now_filename}')
        write_file(output_path_uttblist_idx, context_idxs)
