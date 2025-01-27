#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt

def effective_of_number_samples(num_samples: float) -> float:
    """
    Example placeholder function, as in your original code.
    You can modify or remove this if it’s not needed.
    """
    return num_samples ** 2


def read_file(filepath: str, delimiter: str = ' ') -> List[List[str]]:
    """
    Reads a text file line by line and splits each line by `delimiter`.
    Returns a list of lists: [[token1, token2, ...], [token1, token2, ...], ...].
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split(delimiter) for line in f]


def write_file(filepath: str, data: List[List[str]], delimiter: str = ' ') -> None:
    """
    Writes a list of lists into a text file, joined by `delimiter`.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(delimiter.join(row) + '\n')


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns its contents as a Python object.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def compute_statistics(arr: np.ndarray, label: str = "Array") -> Tuple[float, float, float, float, float]:
    """
    Computes mean, variance, std, min, max and prints them out.
    
    Args:
        arr (np.ndarray): 1D array of numeric values.
        label (str): Optional label for printing.
    
    Returns:
        Tuple of (mean, variance, std, min, max).
    """
    mean_val = np.mean(arr)
    var_val  = np.var(arr)
    std_val  = np.std(arr)
    min_val  = np.min(arr)
    max_val  = np.max(arr)
    print(
        f"{label}: mean={mean_val:.4f}, var={var_val:.4f}, "
        f"std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
    )
    return mean_val, var_val, std_val, min_val, max_val


def normalize(data: List[float]) -> np.ndarray:
    """
    Normalizes a list of numbers to the [0,1] range.
    
    Args:
        data (List[float]): List of numeric values.
    
    Returns:
        np.ndarray: Array of normalized values in [0,1].
    """
    arr = np.array(data, dtype=float)
    arr_min, arr_max = arr.min(), arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def compute_corrected_target_dist(
    raw_target_dist: torch.Tensor, 
    confusion_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Computes the corrected target distribution using BBSE. Specifically:
      p_T(c) = (Q)^{-1} * p_T(hat(c)).

    Args:
        raw_target_dist (torch.Tensor): (C,) tensor of raw probabilities over classes.
        confusion_matrix (torch.Tensor): (C, C) confusion matrix Q, where rows index predicted class
                                         and columns index the true class. Each column sums to 1.

    Returns:
        torch.Tensor: (C,) tensor with the corrected target distribution.
    """
    alpha = 1e-8
    # Suppose 'confusion_matrix' has shape (C, C)
    confusion_matrix_reg = confusion_matrix + alpha * torch.eye(confusion_matrix.shape[0])
    # Invert Q
    Q_t_inv = torch.linalg.inv(confusion_matrix_reg)  # shape (C, C)
    # Correct the distribution
    corrected_dist = Q_t_inv @ raw_target_dist  # shape (C,)
    # Clamp and re-normalize to ensure valid probabilities
    corrected_dist = torch.clamp(
        corrected_dist, 
        min=raw_target_dist.min().item(), 
        max=raw_target_dist.max().item()
    )
    corrected_dist /= corrected_dist.sum()
    return corrected_dist

def main() -> None:
    # File paths (replace with your own as needed)
    context_path              = './local/contextual/contexts/context_f65536_train.txt'
    source_occurrence_path    = './local/contextual/contexts/context_f65536_train_occurrence_train.txt'
    target_occurrence_path    = './local/contextual/contexts/context_f65536_train_occurrence_test_hyp.txt'
    target_gt_occurrence_path = './local/contextual/contexts/context_f65536_train_occurrence_test.txt'
    confusion_matrix_path     = './exp/asr_conformer/run_context_adapter_encoder_suffix/debug/valid/confusion_matrix_S95_sp_final.json'

    context_filename = context_path.split('/')[-1].replace('.txt', '')

    # Read context words
    contexts = [row[0] for row in read_file(context_path)]
    
    # Read source and target counts
    source_counts    = [int(row[0]) for row in read_file(source_occurrence_path)[:-1]]
    target_counts    = [int(row[0]) for row in read_file(target_occurrence_path)[:-1]]
    target_gt_counts = [int(row[0]) for row in read_file(target_gt_occurrence_path)[:-1]]

    # Compute smoothed probabilities for the target domain
    source_probs_dict    = compute_probs(contexts, source_counts)
    target_probs_dict    = compute_probs(contexts, target_counts)
    target_gt_probs_dict = compute_probs(contexts, target_gt_counts)
    
    # Build a consistent index of contexts
    target_contexts = list(target_probs_dict.keys())
    n_classes = len(contexts)
    
    # Initialize confusion matrix
    confusion_matrix_np = np.zeros((n_classes, n_classes), dtype=float)
    np.fill_diagonal(confusion_matrix_np, 1.0)  # Set diagonal to 1.0
    
    # Read raw confusion data from JSON
    data_conf_mat = read_json(confusion_matrix_path)
    for key, count in data_conf_mat.items():
        ground_truth, predicted = key.split('_')
        if '<no-context>' in [ground_truth, predicted]:
            continue
        i = contexts.index(ground_truth)
        j = contexts.index(predicted)
        confusion_matrix_np[i, j] = count
    
    # Create PyTorch tensors
    source_dist      = torch.tensor([source_probs_dict[c] for c in contexts], dtype=torch.float)
    raw_target_dist  = torch.tensor([target_probs_dict[c] for c in target_contexts], dtype=torch.float)
    gt_target_dist   = torch.tensor([target_gt_probs_dict[c] for c in target_contexts], dtype=torch.float)
    confusion_matrix = torch.tensor(confusion_matrix_np, dtype=torch.float)
    
    # Normalize confusion matrix columns via softmax along the last dimension, then transpose
    # (So columns index the true class, and rows the predicted class, each column summing to 1.)
    confusion_matrix   = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True)
    confusion_matrix_t = confusion_matrix.T
    
    # BBSE correction
    output_dir = "./exp/statistics/context_confusion"
    os.makedirs(output_dir, exist_ok=True)
    corrected_target_dist = compute_corrected_target_dist(raw_target_dist, confusion_matrix_t)
    plt.figure(figsize=(10, 6))
    plt.plot(raw_target_dist, alpha=0.5, label='Raw Target Distribution')
    plt.plot(gt_target_dist, alpha=0.5, label='GT Target Distribution')
    plt.plot(corrected_target_dist, alpha=0.5, label='Corrected Target Distribution')
    plt.legend()
    plt.xlabel('Context Index')
    plt.ylabel('Probability')
    plt.title('Raw vs. Corrected Target Distribution')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'target_dist_comparison.png')
    plt.savefig(output_path)

    # Compute statistics
    source_dist_np = source_dist.detach().cpu().numpy()
    corrected_target_dist_np = corrected_target_dist.detach().cpu().numpy()
    compute_statistics(source_dist_np, "Source Distribution")
    compute_statistics(corrected_target_dist_np, "Corrected Target Distribution")

    importance_weights = corrected_target_dist_np / source_dist_np
    compute_statistics(importance_weights, "Importance Weights")
    output_dir = './local/contextual/contexts'
    output_path = os.path.join(output_dir, f'{context_filename}_bbse_importance_weights')
    write_file(
        output_path, 
        [[str(iw)] for iw in importance_weights]
    )

if __name__ == '__main__':
    main()
