import os
import jieba

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyscripts.contextual.utils.dataio import read_file, write_file

if __name__ == '__main__':
    path = './dump/raw/dev'
    text_path = os.path.join(path, 'text')
    context_path = './local/contextual/contexts/context_keywords_dev.txt'
    uid2context_path = './local/contextual/metadata/related_files/dev/keywords'
    # Set the number of worker processes
    now_filename = 'keywords'

    uids = [d[0] for d in read_file(text_path, sp=' ')]
    context = [b[0] for b in read_file(context_path, sp=' ')]
    word2idx = {word: str(i) for i, word in enumerate(context)}
    uid2context = [[d[0].replace('-', '_'), [k for k in d[1].split('$')]] for d in read_file(uid2context_path, sp=' ')]

    print(f'uids: {uids[:5]}')
    print(f'context: {context[:5]}')
    print(f'uid2context: {uid2context[:5]}')

    uid2context_idx = {uid: [word2idx[word] for word in context_words if word in word2idx] for uid, context_words in uid2context}
    uid2context     = {uid: [word for word in context_words if word in word2idx] for uid, context_words in uid2context}
    print(f'uid2context_idx: {uid2context_idx}')

    context_datas = [[uid, " ".join(uid2context[uid])] for uid in uids]
    context_idxs  = [[uid, " ".join(uid2context_idx[uid])] for uid in uids]

    output_path_uttcontext = os.path.join(path, f'uttblist_{now_filename}')
    write_file(output_path_uttcontext, context_datas)

    output_path_uttcontext_idx = os.path.join(path, f'uttblist_idx_{now_filename}')
    write_file(output_path_uttcontext_idx, context_idxs)
