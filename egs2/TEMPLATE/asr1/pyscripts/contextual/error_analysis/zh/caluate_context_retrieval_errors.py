import os
import argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
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

    for relevant_items, hyp_contexts, hyp_probs in zip(ref_context_datas, hyp_context_datas, hyp_context_prob_datas):
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

def is_chinese(word):
    """Check if a word contains Chinese characters."""
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def is_english(word):
    """Check if a word contains English letters."""
    for ch in word:
        if 'a' <= ch.lower() <= 'z':
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate information retrieval metrics including Macro-Averaged Precision, Recall, and F1 at different thresholds.")
    parser.add_argument('--context_list_path', type=str, required=True, help='Path to context list file.')
    parser.add_argument('--ref_context_path', type=str, required=True, help='Path to reference context file.')
    parser.add_argument('--hyp_context_path', type=str, required=True, help='Path to hypothesis context file.')
    parser.add_argument('--hyp_context_prob_path', type=str, required=True, help='Path to hypothesis context probability file.')
    parser.add_argument('--context_candidate_path', type=str, required=True, help='Path to context candidate file.')
    parser.add_argument('--k', type=int, default=5, help='Value of K for metrics.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold.')
    args = parser.parse_args()
    k     = args.k
    thres = args.threshold

    context_list_datas     = [d[0] for d in read_file(args.context_list_path, sp=' ')]
    ref_context_datas      = [list(map(int, filter_space(d[1:]))) for d in read_file(args.ref_context_path, sp=' ')]
    hyp_context_datas      = [filter_space(d[1:]) for d in read_file(args.hyp_context_path, sp=' ')]
    hyp_context_prob_datas = [list(map(float, filter_space(d[1:]))) for d in read_file(args.hyp_context_prob_path, sp=' ')]
    context_candidate_datas = [filter_space(d[1:]) for d in read_file(args.context_candidate_path, sp=' ')]

    ref_context_datas      = [list(map(lambda x: context_list_datas[x], d)) for d in ref_context_datas]

    all_context_words = set()
    for candidates in context_candidate_datas:
        all_context_words.update(candidates)
    all_context_words = list(all_context_words)

    sorted_hyp_context_datas = []
    sorted_hyp_context_prob_datas = []
    for hyp_contexts, hyp_probs in zip(hyp_context_datas, hyp_context_prob_datas):
        hyp_contexts_probs = sorted(zip(hyp_contexts, hyp_probs), key=lambda x: x[1], reverse=True)
        sorted_hyp_contexts = [ctx for ctx, prob in hyp_contexts_probs]
        sorted_hyp_probs = [prob for ctx, prob in hyp_contexts_probs]
        sorted_hyp_context_datas.append(sorted_hyp_contexts)
        sorted_hyp_context_prob_datas.append(sorted_hyp_probs)

    map_score = mean_average_precision(ref_context_datas, sorted_hyp_context_datas, k)
    mrr_score = mean_reciprocal_rank(ref_context_datas, sorted_hyp_context_datas)
    ndcg_score = mean_ndcg(ref_context_datas, sorted_hyp_context_datas, k)
    precision_k = mean_precision_at_k(ref_context_datas, sorted_hyp_context_datas, k)
    mean_precision, mean_recall, mean_f1 = macro_precision_recall_f1_at_threshold(
        ref_context_datas,
        sorted_hyp_context_datas,
        sorted_hyp_context_prob_datas,
        all_context_words,
        threshold=thres 
    )

    global_y_true = []
    global_y_scores = []
    for relevant_items, hyp_contexts, hyp_probs in zip(ref_context_datas, hyp_context_datas, hyp_context_prob_datas):
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))
        for word in all_context_words:
            y_true = 1 if word in relevant_items else 0
            y_score = hyp_context_prob_dict.get(word, 0.0)
            global_y_true.append(y_true)
            global_y_scores.append(y_score)

    if len(np.unique(global_y_true)) > 1:
        roc_auc = roc_auc_score(global_y_true, global_y_scores)
    else:
        roc_auc = None

    print(f"Overall Metrics:")
    print(f"Mean Average Precision at {k}: {map_score:.4f}")
    print(f"Mean Reciprocal Rank: {mrr_score:.4f}")
    print(f"Mean NDCG at {k}: {ndcg_score:.4f}")
    print(f"Mean Precision at {k}: {precision_k:.4f}")
    print(f"Macro-Averaged Precision at threshold {thres}: {mean_precision:.4f}")
    print(f"Macro-Averaged Recall at threshold {thres}: {mean_recall:.4f}")
    print(f"Macro-Averaged F1 Score at threshold {thres}: {mean_f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC (Overall): {roc_auc:.4f}")
    else:
        print("ROC AUC (Overall): Not computable due to lack of class variety.")

    all_chinese_context_words = [word for word in all_context_words if is_chinese(word)]
    all_english_context_words = [word for word in all_context_words if is_english(word)]

    ref_context_datas_chinese = []
    ref_context_datas_english = []

    for ref_context in ref_context_datas:
        ref_chinese = [word for word in ref_context if is_chinese(word)]
        ref_english = [word for word in ref_context if is_english(word)]
        ref_context_datas_chinese.append(ref_chinese)
        ref_context_datas_english.append(ref_english)

    hyp_context_datas_chinese = []
    hyp_context_datas_english = []
    hyp_context_prob_datas_chinese = []
    hyp_context_prob_datas_english = []

    for hyp_contexts, hyp_probs in zip(hyp_context_datas, hyp_context_prob_datas):
        hyp_chinese = []
        hyp_chinese_probs = []
        hyp_english = []
        hyp_english_probs = []
        for word, prob in zip(hyp_contexts, hyp_probs):
            if is_chinese(word):
                hyp_chinese.append(word)
                hyp_chinese_probs.append(prob)
            elif is_english(word):
                hyp_english.append(word)
                hyp_english_probs.append(prob)
        hyp_context_datas_chinese.append(hyp_chinese)
        hyp_context_prob_datas_chinese.append(hyp_chinese_probs)
        hyp_context_datas_english.append(hyp_english)
        hyp_context_prob_datas_english.append(hyp_english_probs)

    sorted_hyp_context_datas_chinese = []
    sorted_hyp_context_prob_datas_chinese = []
    for hyp_contexts, hyp_probs in zip(hyp_context_datas_chinese, hyp_context_prob_datas_chinese):
        hyp_contexts_probs = sorted(zip(hyp_contexts, hyp_probs), key=lambda x: x[1], reverse=True)
        sorted_hyp_contexts = [ctx for ctx, prob in hyp_contexts_probs]
        sorted_hyp_probs = [prob for ctx, prob in hyp_contexts_probs]
        sorted_hyp_context_datas_chinese.append(sorted_hyp_contexts)
        sorted_hyp_context_prob_datas_chinese.append(sorted_hyp_probs)

    sorted_hyp_context_datas_english = []
    sorted_hyp_context_prob_datas_english = []
    for hyp_contexts, hyp_probs in zip(hyp_context_datas_english, hyp_context_prob_datas_english):
        hyp_contexts_probs = sorted(zip(hyp_contexts, hyp_probs), key=lambda x: x[1], reverse=True)
        sorted_hyp_contexts = [ctx for ctx, prob in hyp_contexts_probs]
        sorted_hyp_probs = [prob for ctx, prob in hyp_contexts_probs]
        sorted_hyp_context_datas_english.append(sorted_hyp_contexts)
        sorted_hyp_context_prob_datas_english.append(sorted_hyp_probs)

    map_score_chinese = mean_average_precision(ref_context_datas_chinese, sorted_hyp_context_datas_chinese, k)
    mrr_score_chinese = mean_reciprocal_rank(ref_context_datas_chinese, sorted_hyp_context_datas_chinese)
    ndcg_score_chinese = mean_ndcg(ref_context_datas_chinese, sorted_hyp_context_datas_chinese, k)
    precision_k_chinese = mean_precision_at_k(ref_context_datas_chinese, sorted_hyp_context_datas_chinese, k)
    mean_precision_chinese, mean_recall_chinese, mean_f1_chinese = macro_precision_recall_f1_at_threshold(
        ref_context_datas_chinese,
        sorted_hyp_context_datas_chinese,
        sorted_hyp_context_prob_datas_chinese,
        all_chinese_context_words,
        threshold=thres
    )

    map_score_english = mean_average_precision(ref_context_datas_english, sorted_hyp_context_datas_english, k)
    mrr_score_english = mean_reciprocal_rank(ref_context_datas_english, sorted_hyp_context_datas_english)
    ndcg_score_english = mean_ndcg(ref_context_datas_english, sorted_hyp_context_datas_english, k)
    precision_k_english = mean_precision_at_k(ref_context_datas_english, sorted_hyp_context_datas_english, k)
    mean_precision_english, mean_recall_english, mean_f1_english = macro_precision_recall_f1_at_threshold(
        ref_context_datas_english,
        sorted_hyp_context_datas_english,
        sorted_hyp_context_prob_datas_english,
        all_english_context_words,
        threshold=thres
    )

    global_y_true_chinese = []
    global_y_scores_chinese = []
    for relevant_items, hyp_contexts, hyp_probs in zip(ref_context_datas_chinese, hyp_context_datas_chinese, hyp_context_prob_datas_chinese):
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))
        for word in all_chinese_context_words:
            y_true = 1 if word in relevant_items else 0
            y_score = hyp_context_prob_dict.get(word, 0.0)
            global_y_true_chinese.append(y_true)
            global_y_scores_chinese.append(y_score)

    if len(np.unique(global_y_true_chinese)) > 1:
        roc_auc_chinese = roc_auc_score(global_y_true_chinese, global_y_scores_chinese)
    else:
        roc_auc_chinese = None

    global_y_true_english = []
    global_y_scores_english = []
    for relevant_items, hyp_contexts, hyp_probs in zip(ref_context_datas_english, hyp_context_datas_english, hyp_context_prob_datas_english):
        hyp_context_prob_dict = dict(zip(hyp_contexts, hyp_probs))
        for word in all_english_context_words:
            y_true = 1 if word in relevant_items else 0
            y_score = hyp_context_prob_dict.get(word, 0.0)
            global_y_true_english.append(y_true)
            global_y_scores_english.append(y_score)

    if len(np.unique(global_y_true_english)) > 1:
        roc_auc_english = roc_auc_score(global_y_true_english, global_y_scores_english)
    else:
        roc_auc_english = None

    print(f"\nMetrics for Chinese Context Words:")
    print(f"Mean Average Precision at {k} (Chinese): {map_score_chinese:.4f}")
    print(f"Mean Reciprocal Rank (Chinese): {mrr_score_chinese:.4f}")
    print(f"Mean NDCG at {k} (Chinese): {ndcg_score_chinese:.4f}")
    print(f"Mean Precision at {k} (Chinese): {precision_k_chinese:.4f}")
    print(f"Macro-Averaged Precision at threshold {thres} (Chinese): {mean_precision_chinese:.4f}")
    print(f"Macro-Averaged Recall at threshold {thres} (Chinese): {mean_recall_chinese:.4f}")
    print(f"Macro-Averaged F1 Score at threshold {thres} (Chinese): {mean_f1_chinese:.4f}")
    if roc_auc_chinese is not None:
        print(f"ROC AUC (Chinese): {roc_auc_chinese:.4f}")
    else:
        print("ROC AUC (Chinese): Not computable due to lack of class variety.")

    print(f"\nMetrics for English Context Words:")
    print(f"Mean Average Precision at {k} (English): {map_score_english:.4f}")
    print(f"Mean Reciprocal Rank (English): {mrr_score_english:.4f}")
    print(f"Mean NDCG at {k} (English): {ndcg_score_english:.4f}")
    print(f"Mean Precision at {k} (English): {precision_k_english:.4f}")
    print(f"Macro-Averaged Precision at threshold {thres} (English): {mean_precision_english:.4f}")
    print(f"Macro-Averaged Recall at threshold {thres} (English): {mean_recall_english:.4f}")
    print(f"Macro-Averaged F1 Score at threshold {thres} (English): {mean_f1_english:.4f}")
    if roc_auc_english is not None:
        print(f"ROC AUC (English): {roc_auc_english:.4f}")
    else:
        print("ROC AUC (English): Not computable due to lack of class variety.")

    # thresholds = np.arange(0.0, 1.01, 0.1)

    # print("\nThreshold\tPrecision (Chinese)\tRecall (Chinese)\tF1 Score (Chinese)")
    # for thresh in thresholds:
    #     mean_precision, mean_recall, mean_f1 = macro_precision_recall_f1_at_threshold(
    #         ref_context_datas_chinese,
    #         sorted_hyp_context_datas_chinese,
    #         sorted_hyp_context_prob_datas_chinese,
    #         all_chinese_context_words,
    #         threshold=thresh
    #     )
    #     print(f"{thresh:.1f}\t\t{mean_precision:.4f}\t\t\t{mean_recall:.4f}\t\t\t{mean_f1:.4f}")

    # print("\nThreshold\tPrecision (English)\tRecall (English)\tF1 Score (English)")
    # for thresh in thresholds:
    #     mean_precision, mean_recall, mean_f1 = macro_precision_recall_f1_at_threshold(
    #         ref_context_datas_english,
    #         sorted_hyp_context_datas_english,
    #         sorted_hyp_context_prob_datas_english,
    #         all_english_context_words,
    #         threshold=thresh
    #     )
    #     print(f"{thresh:.1f}\t\t{mean_precision:.4f}\t\t\t{mean_recall:.4f}\t\t\t{mean_f1:.4f}")
