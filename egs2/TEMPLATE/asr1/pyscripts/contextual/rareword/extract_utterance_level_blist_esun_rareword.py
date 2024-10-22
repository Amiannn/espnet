import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyscripts.contextual.utils.dataio import read_file, write_file

occurrence = 2 ** 16
TRAIN_DEV_BLIST_PATH = f"./local/contextual/rarewords/rareword_f{occurrence}_train.txt"
test_occurrence = 10
TEST_BLIST_PATH = f"./local/contextual/rarewords/rareword_f{test_occurrence}_test.txt"

def init_worker(init_bl, init_w2i):
    """Initializer function for each worker in the pool."""
    global blist
    global word2idx
    blist = init_bl
    word2idx = init_w2i

def get_uttblist(words):
    """Process words to get list of rare words and their indices."""
    return [[str(word2idx[word]), word] for word in words if word in blist]

def process_data(data):
    """Function to process each data entry."""
    uttid = data[0]
    results = get_uttblist(data[1:])
    uttblist = [d[1] for d in results]
    uttblist_idx = [d[0] for d in results]
    rareword_data = [uttid] + (uttblist if uttblist else [''])
    rareword_idx = [uttid] + (uttblist_idx if uttblist_idx else [''])
    return rareword_data, rareword_idx

if __name__ == '__main__':
    datas_path = './dump/raw'
    # Set the number of worker processes
    num_workers = cpu_count()  # Use all available CPU cores
    # num_workers = 4          # Or set to a specific number
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' in path:
            blist_path = TEST_BLIST_PATH
            now_occurrence = test_occurrence
        else:
            blist_path = TRAIN_DEV_BLIST_PATH
            now_occurrence = occurrence
        blist = [b[0] for b in read_file(blist_path, sp=' ')]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'Processing {path}...')
        text_path = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')

        with Pool(processes=num_workers, initializer=init_worker, initargs=(blist, word2idx)) as pool:
            results = list(tqdm(pool.imap(process_data, text_datas), total=len(text_datas)))

        rareword_datas, rareword_idxs = zip(*results)
        rareword_datas = list(rareword_datas)
        rareword_idxs = list(rareword_idxs)

        output_path_uttblist = os.path.join(path, f'uttblist_f{now_occurrence}')
        write_file(output_path_uttblist, rareword_datas)

        output_path_uttblist_idx = os.path.join(path, f'uttblist_idx_f{now_occurrence}')
        write_file(output_path_uttblist_idx, rareword_idxs)
