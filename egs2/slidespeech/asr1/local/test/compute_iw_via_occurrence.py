import os
import numpy as np

context_path            = './local/contextual/contexts/context_f65536_train.txt'
target_context_idx_path = './dump/raw/S95_sp/uttblist_idx_f65536'
source_occurrence_path  = './local/contextual/contexts/context_f65536_train_occurrence_train.txt'
target_occurrence_path  = './local/contextual/contexts/context_f65536_train_occurrence_test.txt'

context_filename = context_path.split('/')[-1].replace('.txt', '')

def effective_of_number_samples(c):
    # beta = 0.8
    # return (1 - beta**c) / (1 - beta)
    return  c**2

def read_file(path, sp=' '):
    """
    Reads a text file line by line and splits each line by `sp`.
    Returns a list of lists: [[uid, token1, token2, ...], ...].
    """
    with open(path, 'r') as f:
        return [line.strip().split(sp) for line in f]

def write_file(path, datas, sp=' '):
    """
    Writes a list of tokens into a text file, joined by `sp`.
    """
    with open(path, 'w') as f:
        for data in datas:
            f.write(sp.join(data) + '\n')

def compute_importance_weights(words_a, counts_a, words_b, counts_b):
    alpha = 1e-6

    counts_a_dict = dict((wa, effective_of_number_samples(float(ca))) for wa, ca in zip(words_a, counts_a))
    counts_b_dict = dict((wb, effective_of_number_samples(float(cb))) for wb, cb in zip(words_b, counts_b))

    sum_counts_a = sum(counts_a_dict.values())
    sum_counts_b = sum(counts_b_dict.values())

    vocab_a = set(counts_a_dict.keys())
    vocab_b = set(counts_b_dict.keys())

    len_vocab_a = len(vocab_a)
    len_vocab_b = len(vocab_b)

    all_words = vocab_a | vocab_b

    freq_a = {}
    freq_b = {}
    for w in all_words:
        ca = counts_a_dict.get(w, 0)
        cb = counts_b_dict.get(w, 0)
        # Smoothed frequencies
        freq_a[w] = (ca + alpha) / (sum_counts_a + alpha * len_vocab_a)
        freq_b[w] = (cb + alpha) / (sum_counts_b + alpha * len_vocab_b)

    importance_weights = {}
    for w in all_words:
        importance_weights[w] = freq_b[w] / freq_a[w]
        # importance_weights[w] = 1 / freq_a[w]
    return importance_weights

def compute_statistics(arr, label="Array"):
    """
    Computes mean, variance, std, min, max and prints them out.
    Returns them in a tuple for convenience.
    """
    mean_val = np.mean(arr)
    var_val  = np.var(arr)
    std_val  = np.std(arr)
    min_val  = np.min(arr)
    max_val  = np.max(arr)
    print(f"{label}: mean={mean_val:.4f}, var={var_val:.4f}, std={std_val:.4f}, "
          f"min={min_val:.4f}, max={max_val:.4f}")
    return mean_val, var_val, std_val, min_val, max_val

def normalize(datas):
    datas = np.array(datas)
    iw_max = np.max(datas)
    iw_min = np.min(datas)
    return (datas - iw_min) / (iw_max - iw_min)

contexts           = [d[0] for d in read_file(context_path)]
target_context_idx = [[d[0], [int(idx) for idx in d[1:]]] for d in read_file(target_context_idx_path)]
source_occurrences = [int(d[0]) for d in read_file(source_occurrence_path)]
target_occurrences = [int(d[0]) for d in read_file(target_occurrence_path)]

context_idxs = list(range(len(contexts)))
importance_weights = compute_importance_weights(
    context_idxs, 
    source_occurrences,
    context_idxs,
    target_occurrences
)

output_dir = './local/contextual/contexts'
output_path = os.path.join(output_dir, f'{context_filename}_importance_weights')
write_file(
    output_path, 
    [[str(importance_weights[idx])] for idx in importance_weights]
)

datas = []
utterance_iws = []
for uid, idxs in target_context_idx:
    iws = 0
    for idx in idxs:
        datas.append(np.log(importance_weights[idx]))
        iws += np.log(importance_weights[idx])
    utterance_iws.append((iws / len(idxs)) if len(idxs) > 0 else 0)

compute_statistics(utterance_iws, "IW")
output_dir = './exp/statistics/context_importance_weights'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'importance_weights.txt')

utterance_iws = normalize(utterance_iws)
utterance_iws = [[str(d)] for d in utterance_iws]
write_file(output_path, utterance_iws)