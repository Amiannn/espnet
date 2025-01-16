#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import random

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import sentencepiece as spm

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ESPnet imports
from espnet2.tasks.lm import LMTask
from espnet2.torch_utils.device_funcs import to_device

###############################################################################
#                               UTILITY FUNCTIONS
###############################################################################
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

###############################################################################
#                    LANGUAGE MODEL LOG-PROBABILITY COMPUTATION
###############################################################################
def get_word_log_prob(model, tokenizer, text):
    """
    Given a text, compute its average log probability per word under the provided LM.
    
    Args:
      model:      The trained LM model (with .lm(x, None) method).
      tokenizer:  A SentencePieceProcessor (or any tokenizer) that supports .encode().
      text:       A string whose log probability is to be computed.
    
    Returns:
      avg_log_prob (float): The average log probability across words.
    """
    # Step 1: Convert text to uppercase and split into words
    words = text.upper().split(" ")
    
    # Step 2: Encode each word into tokens
    # This will result in a list of lists, where each sublist contains token IDs for a word
    word_token_lists = [tokenizer.encode(word) for word in words]
    
    # Step 3: Flatten the list of tokens and keep track of word boundaries
    tokens = [model.sos]  # Start with the start-of-sentence token
    token_word_mapping = []  # This will map each token to its corresponding word index
    
    for word_idx, word_tokens in enumerate(word_token_lists):
        tokens.extend(word_tokens)  # Add tokens for the current word
        token_word_mapping.extend([word_idx] * len(word_tokens))  # Map tokens to the current word
    tokens.append(model.eos)  # Add the end-of-sentence token
    token_word_mapping += [len(word_token_lists)]
    # Convert tokens to a tensor and add batch dimension
    x = torch.tensor(tokens).unsqueeze(0)  # Shape: (1, sequence_length)
    
    with torch.no_grad():
        # Step 4: Pass tokens through the language model to get logits
        output, _ = model.lm(x, None)  # Output shape: (1, sequence_length, vocab_size)
        
        # Compute log probabilities using log_softmax
        log_probs = torch.log_softmax(output.squeeze(0), dim=-1)  # Shape: (sequence_length, vocab_size)
        
        # Initialize a list to accumulate log probabilities per word
        word_log_probs = [0.0 for _ in words] + [0.0]
        
        word_token_lists += [[model.eos]]
        # Iterate over each position in the token sequence (excluding the last token)
        for t in range(len(tokens) - 1):
            target_token_id = tokens[t + 1]  # The token we want the probability for
            word_idx = token_word_mapping[t]  # Which word this token belongs to
            word_log_probs[word_idx] += log_probs[t, target_token_id].item() / len(word_token_lists[word_idx])
        
    # Step 5: Compute the average log probability across all words
    avg_log_prob = sum(word_log_probs) / len(words)
    # avg_log_prob = sum(word_log_probs)
    
    return avg_log_prob

def compute_log_probs_for_dataset(model, tokenizer, data, max_utt=None):
    """
    Computes log probabilities for each utterance in `data` with a given model.
    
    Args:
      model:      Trained LM model.
      tokenizer:  A SentencePieceProcessor for tokenization.
      data:       List of (uid, text) pairs.
      max_utt:    (Optional) Limit on number of utterances to process.

    Returns:
      A list of float log probabilities, one per utterance in `data`.
    """
    log_probs = []
    if max_utt is not None:
        data = data[:max_utt]

    for uid, text in tqdm(data, desc=f"Computing log probabilities with {model.__class__.__name__}"):
        lp = get_word_log_prob(model, tokenizer, text)
        log_probs.append(lp)
    return log_probs

###############################################################################
#                             PLOTTING AND STATISTICS
###############################################################################
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

def plot_scatter(
    x, y, 
    x_label="X",
    y_label="Y",
    title="Scatter Plot",
    save_path=None
):
    """
    Plot a scatter plot (with regression line) given two arrays x and y.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"

    plt.figure(figsize=(8, 6))
    sns.regplot(
        x=x,
        y=y,
        ci=95,
        scatter_kws={'alpha': 0.6, 'edgecolor': None},
        line_kws={'color': 'red', 'lw': 2}
    )
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_histogram(
    data,
    bins=50,
    kde=True,
    color='blue',
    title="Histogram",
    x_label="Values",
    y_label="Count",
    save_path=None
):
    """
    Plot a histogram (optionally with KDE) of `data`.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"

    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=bins, kde=kde, color=color)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

###############################################################################
#           NEW: FUNCTION TO PLOT OVERLAPPING HISTOGRAMS / KDEs
###############################################################################
def plot_overlapping_histograms(
    data_list,
    labels,
    bins=50,
    alpha=0.5,
    colors=None,
    title="Overlapping Histograms",
    x_label="Values",
    y_label="Count",
    save_path=None
):
    """
    Plot multiple histograms (overlapped) on one figure for easier comparison.
    
    Args:
        data_list (list of arrays): List of numeric arrays to plot.
        labels (list of str): List of labels corresponding to each array.
        bins (int): Number of histogram bins.
        alpha (float): Transparency for each histogram.
        colors (list of str): List of colors (e.g. ['blue', 'red']).
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        save_path (str): Path to save the figure. If None, does not save.
    """
    if colors is None:
        # Provide a default list of colors if none given
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"

    plt.figure(figsize=(8, 6))

    for i, data in enumerate(data_list):
        label = labels[i] if i < len(labels) else f"Series {i}"
        color = colors[i] if i < len(colors) else None
        sns.histplot(data, bins=bins, kde=False, color=color, label=label, alpha=alpha)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_overlapping_kde(
    data_list,
    labels,
    colors=None,
    alpha=0.7,
    title="Overlapping KDE Plots",
    x_label="Values",
    y_label="Density",
    save_path=None
):
    """
    Plot multiple KDE curves (overlapped) on one figure for easier comparison.
    """
    if colors is None:
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"

    plt.figure(figsize=(8, 6))

    for i, data in enumerate(data_list):
        label = labels[i] if i < len(labels) else f"Series {i}"
        color = colors[i] if i < len(colors) else None
        sns.kdeplot(data, fill=True, color=color, label=label, alpha=alpha)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

###############################################################################
#                                   MAIN LOGIC
###############################################################################
def main():
    # ============ 1) SETTINGS ============
    output_dir = "./exp/statistics/context_language_model"
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"

    bpe_path = "./data/en_token_list/bpe_unigram5000suffix/bpe.model"
    tokenizer = spm.SentencePieceProcessor(model_file=bpe_path)

    # Paths to source-domain model
    source_model_file   = "./exp/lm/source_domain/1000epoch.pth"
    source_train_config = "./exp/lm/source_domain/config.yaml"

    # Paths to target-domain model
    target_model_file   = "./exp/lm/target_domain/2000epoch.pth"
    target_train_config = "./exp/lm/target_domain/config.yaml"

    # Load LMs
    source_model, source_train_args = LMTask.build_model_from_file(
        source_train_config, source_model_file, device
    )
    source_model.eval()

    target_model, target_train_args = LMTask.build_model_from_file(
        target_train_config, target_model_file, device
    )
    target_model.eval()

    # ============ 2) READ DATA ============
    # We assume two datasets:
    #   (a) "source_data_path"  - e.g., from source domain
    #   (b) "target_data_path"  - e.g., from target domain
    source_data_path = "./dump/raw/S95_sp/uttblist_f65536_rm_spaces"
    target_data_path = "./dump/raw/test/uttblist_keywords"

    # Convert lines to [(uid, text), ...]
    source_data = [[d[0], " ".join(d[1:]).upper()] for d in read_file(source_data_path, sp=" ")]
    target_data = [[d[0], " ".join(d[1:]).upper()] for d in read_file(target_data_path, sp=" ")]

    # random.shuffle(source_data)
    # random.shuffle(target_data)

    # ============ 3) COMPUTE LOG PROBS (SOURCE DATASET) ============
    print("\n=== SOURCE DATASET ===")
    source_log_probs_on_source = compute_log_probs_for_dataset(
        source_model, tokenizer, source_data, max_utt=1000
    )
    target_log_probs_on_source = compute_log_probs_for_dataset(
        target_model, tokenizer, source_data, max_utt=1000
    )

    # Print basic statistics
    print("--- Statistics on Source Dataset ---")
    compute_statistics(source_log_probs_on_source, "SourceModel(SourceData)")
    compute_statistics(target_log_probs_on_source, "TargetModel(SourceData)")

    # Correlation
    correlation_source = np.corrcoef(source_log_probs_on_source, target_log_probs_on_source)[0, 1]
    print(f"Correlation between SourceModel vs. TargetModel on Source dataset: {correlation_source:.4f}")

    # ============ PLOTS (SOURCE DATASET) ============
    # 3.1) Scatter
    scatter_save_path = os.path.join(output_dir, "scatter_source_data.png")
    plot_scatter(
        x=source_log_probs_on_source, 
        y=target_log_probs_on_source,
        x_label="Source Model Log Prob (on Source Data)",
        y_label="Target Model Log Prob (on Source Data)",
        title="Source vs Target Model on Source Dataset",
        save_path=scatter_save_path
    )

    # 3.2) Histogram of log-differences
    diff_source = np.array(target_log_probs_on_source) - np.array(source_log_probs_on_source)
    print(f"\nLog diff stats (Target - Source on Source data):")
    compute_statistics(diff_source, label="LogDiff(SourceData)")
    hist_save_path = os.path.join(output_dir, "hist_log_diff_source_data.png")
    plot_histogram(
        diff_source,
        bins=50,
        kde=True,
        color='#1f77b4',
        title="Histogram of Log Differences (Target - Source on Source Data)",
        x_label="Log Importance Weights",
        y_label="Count",
        save_path=hist_save_path
    )

    # 3.3) Overlapping histogram of log probabilities from both models
    overlap_hist_source_path = os.path.join(output_dir, "overlap_hist_source_data.png")
    plot_overlapping_histograms(
        data_list=[
            source_log_probs_on_source, 
            target_log_probs_on_source
        ],
        labels=[
            "Source Model on Source Data",
            "Target Model on Source Data"
        ],
        bins=50,
        alpha=0.5,
        colors=["#1f77b4", "#2ca02c"],
        title="Source vs. Target Model (Source Data)",
        x_label="Log Probability",
        y_label="Count",
        save_path=overlap_hist_source_path
    )

    # 3.4) Overlapping KDE
    overlap_kde_source_path = os.path.join(output_dir, "overlap_kde_source_data.png")
    plot_overlapping_kde(
        data_list=[
            source_log_probs_on_source,
            target_log_probs_on_source
        ],
        labels=[
            "Source Model on Source Data",
            "Target Model on Source Data"
        ],
        colors=["#1f77b4", "#2ca02c"],
        alpha=0.6,
        title="Source vs. Target Model (Source Data)",
        x_label="Log Probability",
        y_label="Density",
        save_path=overlap_kde_source_path
    )

    # ============ 4) COMPUTE LOG PROBS (TARGET DATASET) ============
    print("\n=== TARGET DATASET ===")
    source_log_probs_on_target = compute_log_probs_for_dataset(
        source_model, tokenizer, target_data, max_utt=5000
    )
    target_log_probs_on_target = compute_log_probs_for_dataset(
        target_model, tokenizer, target_data, max_utt=5000
    )

    # Print basic statistics
    print("--- Statistics on Target Dataset ---")
    compute_statistics(source_log_probs_on_target, "SourceModel(TargetData)")
    compute_statistics(target_log_probs_on_target, "TargetModel(TargetData)")

    # Correlation
    correlation_target = np.corrcoef(source_log_probs_on_target, target_log_probs_on_target)[0, 1]
    print(f"Correlation between SourceModel vs. TargetModel on Target dataset: {correlation_target:.4f}")

    # ============ PLOTS (TARGET DATASET) ============
    # 4.1) Scatter
    scatter_save_path = os.path.join(output_dir, "scatter_target_data.png")
    plot_scatter(
        x=source_log_probs_on_target, 
        y=target_log_probs_on_target,
        x_label="Source Model Log Prob (on Target Data)",
        y_label="Target Model Log Prob (on Target Data)",
        title="Source vs Target Model on Target Dataset",
        save_path=scatter_save_path
    )

    # 4.2) Histogram of log-differences
    diff_target = np.array(target_log_probs_on_target) - np.array(source_log_probs_on_target)
    print(f"\nLog diff stats (Target - Source on Target data):")
    compute_statistics(diff_target, label="LogDiff(TargetData)")
    hist_save_path = os.path.join(output_dir, "hist_log_diff_target_data.png")
    plot_histogram(
        diff_target,
        bins=50,
        kde=True,
        color='#9467bd',
        title="Histogram of Log Differences (Target - Source on Target Data)",
        x_label="Log Importance Weights",
        y_label="Count",
        save_path=hist_save_path
    )

    # 4.3) Overlapping histogram (Target dataset)
    overlap_hist_target_path = os.path.join(output_dir, "overlap_hist_target_data.png")
    plot_overlapping_histograms(
        data_list=[
            source_log_probs_on_target, 
            target_log_probs_on_target
        ],
        labels=[
            "Source Model on Target Data",
            "Target Model on Target Data"
        ],
        bins=50,
        alpha=0.5,
        colors=["#1f77b4", "#2ca02c"],
        title="Source vs. Target Model (Target Data)",
        x_label="Log Probability",
        y_label="Count",
        save_path=overlap_hist_target_path
    )

    # 4.4) Overlapping KDE (Target dataset)
    overlap_kde_target_path = os.path.join(output_dir, "overlap_kde_target_data.png")
    plot_overlapping_kde(
        data_list=[
            source_log_probs_on_target,
            target_log_probs_on_target
        ],
        labels=[
            "Source Model on Target Data",
            "Target Model on Target Data"
        ],
        colors=["#1f77b4", "#2ca02c"],
        alpha=0.6,
        title="Source vs. Target Model (Target Data)",
        x_label="Log Probability",
        y_label="Density",
        save_path=overlap_kde_target_path
    )

    overlap_kde_diff_source_path = os.path.join(output_dir, "overlap_kde_diff_source_data.png")
    plot_overlapping_kde(
        data_list=[
            diff_source,
            diff_target
        ],
        labels=[
            "Diff on Source Data",
            "Diff on Target Data"
        ],
        colors=["#1f77b4", "#2ca02c"],
        alpha=0.6,
        title="Source vs. Target Data (Log Importance Weight)",
        x_label="Log Probability",
        y_label="Density",
        save_path=overlap_kde_diff_source_path
    )

    output_path = os.path.join(output_dir, "importance_weights.txt")
    diff_target = [[str(diff)] for diff in diff_target]
    write_file(output_path, diff_target)

    print("\nDone. All plots saved in:", output_dir)


###############################################################################
#                                ENTRY POINT
###############################################################################
if __name__ == "__main__":
    # If you want CLI args, you can expand parse_args. Otherwise just call main:
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    main()
