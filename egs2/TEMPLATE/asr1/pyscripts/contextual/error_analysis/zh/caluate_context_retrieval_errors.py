import os
import argparse
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    roc_curve,
    precision_score,
    recall_score
)

def filter_space(data):
    return [d for d in data if d != '' and d != '⁇']

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def average_precision_at_k(relevant_items, retrieved_items, k):
    """Compute Average Precision at K for a single query"""
    relevant_set = set(relevant_items)
    retrieved_list = retrieved_items[:k]
    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(retrieved_list):
        if item in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not relevant_items:
        return 1.0
    return score / min(len(relevant_items), k)

def mean_average_precision(relevant_list, retrieved_list, k):
    """Compute Mean Average Precision at K over all queries"""
    avg_precisions = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        avg_prec = average_precision_at_k(relevant_items, retrieved_items, k)
        avg_precisions.append(avg_prec)
    return np.mean(avg_precisions)

def reciprocal_rank(relevant_items, retrieved_items):
    """Compute Reciprocal Rank for a single query"""
    relevant_set = set(relevant_items)
    for i, item in enumerate(retrieved_items):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(relevant_list, retrieved_list):
    """Compute Mean Reciprocal Rank over all queries"""
    rr_scores = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        rr = reciprocal_rank(relevant_items, retrieved_items)
        rr_scores.append(rr)
    return np.mean(rr_scores)

def dcg_at_k(relevant_items, retrieved_items, k):
    """Compute Discounted Cumulative Gain at K for a single query"""
    relevant_set = set(relevant_items)
    dcg = 0.0
    for i, item in enumerate(retrieved_items[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def idcg_at_k(relevant_items, k):
    """Compute Ideal Discounted Cumulative Gain at K for a single query"""
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    return idcg

def ndcg_at_k(relevant_items, retrieved_items, k):
    """Compute Normalized Discounted Cumulative Gain at K for a single query"""
    dcg = dcg_at_k(relevant_items, retrieved_items, k)
    idcg = idcg_at_k(relevant_items, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def mean_ndcg(relevant_list, retrieved_list, k):
    """Compute Mean NDCG at K over all queries"""
    ndcg_scores = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        ndcg = ndcg_at_k(relevant_items, retrieved_items, k)
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def precision_at_k(relevant_items, retrieved_items, k):
    """Compute Precision at K for a single query"""
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_items[:k])
    return len(relevant_set & retrieved_set) / k

def mean_precision_at_k(relevant_list, retrieved_list, k):
    """Compute Mean Precision at K over all queries"""
    precisions = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        prec = precision_at_k(relevant_items, retrieved_items, k)
        precisions.append(prec)
    return np.mean(precisions)

def precision_recall_f1_per_query(relevant_items, retrieved_items):
    """Compute Precision, Recall, and F1 for a single query"""
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_items)
    # print(f'relevant_set: {relevant_set}')
    # print(f'retrieved_set: {retrieved_set}')
    # print(f'_' * 30)
    tp = len(relevant_set & retrieved_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

def macro_precision_recall_f1_at_threshold(ref_context_datas, hyp_context_datas, hyp_context_prob_datas, all_context_words, threshold):
    """Compute Macro-Averaged Precision, Recall, and F1 at a given threshold"""
    precisions = []
    recalls = []
    f1_scores = []

    for i, (relevant_items, hyp_contexts, hyp_probs) in enumerate(zip(ref_context_datas, hyp_context_datas, hyp_context_prob_datas)):
        # Create a dictionary for quick lookup
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))

        # Determine which words are predicted positive at this threshold
        retrieved_items = [word for word in all_context_words if hyp_context_prob_dict.get(word, 0.0) >= threshold]

        # Compute Precision, Recall, F1 for this query
        precision, recall, f1 = precision_recall_f1_per_query(relevant_items, retrieved_items)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Compute macro-averaged scores
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    return mean_precision, mean_recall, mean_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate information retrieval metrics including Macro-Averaged Precision, Recall, and F1 at different thresholds.")
    parser.add_argument('--context_list_path', type=str, required=True, help='Path to context list file.')
    parser.add_argument('--ref_context_path', type=str, required=True, help='Path to reference context file.')
    parser.add_argument('--hyp_context_path', type=str, required=True, help='Path to hypothesis context file.')
    parser.add_argument('--hyp_context_prob_path', type=str, required=True, help='Path to hypothesis context probability file.')
    parser.add_argument('--context_candidate_path', type=str, required=True, help='Path to context candidate file.')
    parser.add_argument('--k', type=int, default=5, help='Value of K for metrics.')
    args = parser.parse_args()
    k = args.k

    context_list_datas     = [d[0] for d in read_file(args.context_list_path, sp=' ')]
    ref_context_datas      = [list(map(int, filter_space(d[1:]))) for d in read_file(args.ref_context_path, sp=' ')]
    hyp_context_datas      = [filter_space(d[1:]) for d in read_file(args.hyp_context_path, sp=' ')]
    hyp_context_prob_datas = [list(map(float, filter_space(d[1:]))) for d in read_file(args.hyp_context_prob_path, sp=' ')]
    context_candidate_datas = [filter_space(d[1:]) for d in read_file(args.context_candidate_path, sp=' ')]

    ref_context_datas      = [list(map(lambda x: context_list_datas[x], d)) for d in ref_context_datas]

    # Sort the predicted contexts based on probabilities
    sorted_hyp_context_datas = []
    sorted_hyp_context_prob_datas = []
    for hyp_contexts, hyp_probs in zip(hyp_context_datas, hyp_context_prob_datas):
        hyp_contexts_probs = sorted(zip(hyp_contexts, hyp_probs), key=lambda x: x[1], reverse=True)
        sorted_hyp_contexts = [ctx for ctx, prob in hyp_contexts_probs]
        sorted_hyp_probs = [prob for ctx, prob in hyp_contexts_probs]
        sorted_hyp_context_datas.append(sorted_hyp_contexts)
        sorted_hyp_context_prob_datas.append(sorted_hyp_probs)

    # Prepare data for overall ROC AUC computation
    # Collect all unique keywords
    all_context_words = set()
    for candidates in context_candidate_datas:
        all_context_words.update(candidates)
    all_context_words = list(all_context_words)

    # Calculate evaluation metrics
    map_score = mean_average_precision(ref_context_datas, sorted_hyp_context_datas, k)
    mrr_score = mean_reciprocal_rank(ref_context_datas, sorted_hyp_context_datas)
    ndcg_score = mean_ndcg(ref_context_datas, sorted_hyp_context_datas, k)
    precision_k = mean_precision_at_k(ref_context_datas, sorted_hyp_context_datas, k)
    mean_precision, mean_recall, mean_f1 = macro_precision_recall_f1_at_threshold(
        ref_context_datas,
        sorted_hyp_context_datas,
        sorted_hyp_context_prob_datas,
        all_context_words,
        threshold=0.5  # Default threshold for initial metrics
    )

    print(f'Mean Average Precision at {k}: {map_score:.4f}')
    print(f'Mean Reciprocal Rank: {mrr_score:.4f}')
    print(f'Mean NDCG at {k}: {ndcg_score:.4f}')
    print(f'Mean Precision at {k}: {precision_k:.4f}')
    print(f'Macro-Averaged Precision at threshold 0.5: {mean_precision:.4f}')
    print(f'Macro-Averaged Recall at threshold 0.5: {mean_recall:.4f}')
    print(f'Macro-Averaged F1 Score at threshold 0.5: {mean_f1:.4f}')

    # Initialize lists to hold global y_true and y_scores
    global_y_true = []
    global_y_scores = []

    for i, (relevant_items, hyp_contexts, hyp_probs) in enumerate(zip(ref_context_datas, hyp_context_datas, hyp_context_prob_datas)):
        # Create a dictionary for quick lookup
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))
        # For each keyword
        for word in all_context_words:
            # y_true: 1 if word in relevant_items else 0
            y_true = 1 if word in relevant_items else 0
            # y_score: hyp_context_prob_dict.get(word, 0.0)
            y_score = hyp_context_prob_dict.get(word, 0.0)
            global_y_true.append(y_true)
            global_y_scores.append(y_score)

    # Convert lists to numpy arrays
    global_y_true = np.array(global_y_true)
    global_y_scores = np.array(global_y_scores)

    # Step 1: Compute ROC AUC
    if len(np.unique(global_y_true)) > 1:
        roc_auc = roc_auc_score(global_y_true, global_y_scores)
        print(f"Overall ROC AUC: {roc_auc:.4f}")
    else:
        print("Cannot compute ROC AUC because only one class is present in y_true.")

    # Define thresholds from 0 to 1 in steps of 0.1
    thresholds = np.arange(0.0, 1.01, 0.1)

    # print("\nThreshold\tMacro Precision\tMacro Recall\tMacro F1 Score")
    # for thresh in thresholds:
    #     mean_precision, mean_recall, mean_f1 = macro_precision_recall_f1_at_threshold(
    #         ref_context_datas,
    #         sorted_hyp_context_datas,
    #         sorted_hyp_context_prob_datas,
    #         all_context_words,
    #         threshold=thresh
    #     )
    #     print(f"{thresh:.1f}\t\t{mean_precision:.4f}\t\t{mean_recall:.4f}\t\t{mean_f1:.4f}")
