import os
import numpy as np
from typing import List, Tuple, Dict, Any

context_path              = './local/contextual/contexts/context_f65536_train.txt'
source_occurrence_path    = './local/contextual/contexts/context_f65536_train_occurrence_train.txt'
target_occurrence_path    = './local/contextual/contexts/context_f65536_train_occurrence_test.txt'
target_context_idx_path   = './dump/raw/test/uttblist_idx_keywords'

context_filename = context_path.split('/')[-1].replace('.txt', '')

def effective_of_number_samples(num_samples: float) -> float:
    """
    Example placeholder function, as in your original code.
    You can modify or remove this if it’s not needed.
    """
    return num_samples ** 2

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

def compute_probs(
    words: List[str],
    counts: List[int],
    alpha: float = 1e-6
) -> Dict[str, float]:
    """
    Given a list of words and their counts, computes a (smoothed) frequency distribution.
    
    Args:
        words (List[str]): The vocabulary (list of words).
        counts (List[int]): Occurrence counts, parallel to `words`.
        alpha (float): Smoothing parameter for additive smoothing.
    
    Returns:
        Dict[str, float]: Mapping from word -> smoothed probability.
    """
    # Build a dictionary of {word: effective_count}
    counts_dict = {
        w: effective_of_number_samples(float(c))
        for w, c in zip(words, counts)
    }
    
    total_count = sum(counts_dict.values())
    vocab_size = len(counts_dict)

    freqs = {}
    for w in words:
        c_w = counts_dict.get(w, 0.0)
        freqs[w] = (c_w + alpha) / (total_count + alpha * vocab_size)
    return freqs

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
source_dist = compute_probs(context_idxs, source_occurrences)
target_dist = compute_probs(context_idxs, target_occurrences)

source_dist_np = np.array([source_dist[idx] for idx in context_idxs])
target_dist_np = np.array([target_dist[idx] for idx in context_idxs])

importance_weights = target_dist_np / source_dist_np
print(f"Importance weights: {importance_weights}")

output_dir = './local/contextual/contexts'
output_path = os.path.join(output_dir, f'{context_filename}_gt_importance_weights')
write_file(
    output_path, 
    [[str(idx)] for idx in importance_weights]
)