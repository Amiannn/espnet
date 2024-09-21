import os
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

def filter_space(data):
    return [d for d in data if d != '']

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
    """計算單一查詢的 Precision at K"""
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_items[:k])
    return len(relevant_set & retrieved_set) / k

def mean_precision_at_k(relevant_list, retrieved_list, k):
    """計算整個資料集的 Mean Precision at K"""
    precisions = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        prec = precision_at_k(relevant_items, retrieved_items, k)
        precisions.append(prec)
    return np.mean(precisions)

def precision_recall_f1(relevant_items, retrieved_items):
    """Compute Precision, Recall, and F1 for a single query"""
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_items)
    tp = len(relevant_set & retrieved_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def mean_precision_recall_f1(relevant_list, retrieved_list):
    """Compute mean Precision, Recall, and F1 over all queries"""
    precisions = []
    recalls = []
    f1_scores = []
    for relevant_items, retrieved_items in zip(relevant_list, retrieved_list):
        precision, recall, f1 = precision_recall_f1(relevant_items, retrieved_items)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    return mean_precision, mean_recall, mean_f1

def compute_roc_auc(y_true_list, y_score_list):
    """Compute ROC Curve and AUC"""
    y_true = np.concatenate(y_true_list)
    y_scores = np.concatenate(y_score_list)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate information retrieval metrics including F1, Recall, Precision, and ROC.")
    parser.add_argument('--context_list_path', type=str, required=True, help='Path to context list file.')
    parser.add_argument('--ref_context_path', type=str, required=True, help='Path to reference context file.')
    parser.add_argument('--hyp_context_path', type=str, required=True, help='Path to hypothesis context file.')
    parser.add_argument('--hyp_context_prob_path', type=str, required=True, help='Path to hypothesis context probability file.')
    parser.add_argument('--context_candidate_path', type=str, required=True, help='Path to context candidate file.')
    parser.add_argument('--k', type=int, default=5, help='Value of K for metrics.')
    args = parser.parse_args()
    k = args.k

    context_list_datas     = [d[0] for d in read_file(args.context_list_path, sp=' ')]
    ref_context_datas      = [d[1:] for d in read_file(args.ref_context_path, sp=' ')]
    hyp_context_datas      = [filter_space(d[1:]) for d in read_file(args.hyp_context_path, sp=' ')]
    hyp_context_prob_datas = [list(map(float, filter_space(d[1:]))) for d in read_file(args.hyp_context_prob_path, sp=' ')]
    context_candidate_datas = [filter_space(d[1:]) for d in read_file(args.context_candidate_path, sp=' ')]

    # Sort the predicted contexts based on probabilities
    sorted_hyp_context_datas = []
    sorted_hyp_context_prob_datas = []
    for hyp_contexts, hyp_probs in zip(hyp_context_datas, hyp_context_prob_datas):
        hyp_contexts_probs = sorted(zip(hyp_contexts, hyp_probs), key=lambda x: x[1], reverse=True)
        sorted_hyp_contexts = [ctx for ctx, prob in hyp_contexts_probs]
        sorted_hyp_probs = [prob for ctx, prob in hyp_contexts_probs]
        sorted_hyp_context_datas.append(sorted_hyp_contexts)
        sorted_hyp_context_prob_datas.append(sorted_hyp_probs)

    # Calculate evaluation metrics
    map_score = mean_average_precision(ref_context_datas, sorted_hyp_context_datas, k)
    mrr_score = mean_reciprocal_rank(ref_context_datas, sorted_hyp_context_datas)
    ndcg_score = mean_ndcg(ref_context_datas, sorted_hyp_context_datas, k)
    precision_k = mean_precision_at_k(ref_context_datas, sorted_hyp_context_datas, k)
    mean_precision, mean_recall, mean_f1 = mean_precision_recall_f1(ref_context_datas, sorted_hyp_context_datas)

    print(f'Mean Average Precision at {k}: {map_score:.4f}')
    print(f'Mean Reciprocal Rank: {mrr_score:.4f}')
    print(f'Mean NDCG at {k}: {ndcg_score:.4f}')
    print(f'Mean Precision at {k}: {precision_k:.4f}')
    print(f'Mean Precision: {mean_precision:.4f}')
    print(f'Mean Recall: {mean_recall:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')

    # Compute ROC Curve and AUC
    # Prepare true labels and scores for ROC computation
    y_true_list = []
    y_score_list = []
    all_context_words = set()
    for candidates in context_candidate_datas:
        all_context_words.update(candidates)

    all_context_words = list(all_context_words)

    for relevant_items, hyp_contexts, hyp_probs in zip(ref_context_datas, hyp_context_datas, hyp_context_prob_datas):
        y_true = []
        y_scores = []
        # Create a dictionary for quick lookup
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))
        for word in all_context_words:
            y_true.append(1 if word in relevant_items else 0)
            y_scores.append(hyp_context_prob_dict.get(word, 0.0))  # If not predicted, probability is 0.0
        y_true_list.append(y_true)
        y_score_list.append(y_scores)

    fpr, tpr, roc_auc = compute_roc_auc(y_true_list, y_score_list)
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # Optionally, you can plot the ROC Curve
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
