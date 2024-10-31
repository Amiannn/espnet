# Define the array of experiment folder names
exp_folder=(
  "run_medium_dotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_dotproduct_contextual_retriever_suffix"
  "run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_xdotproduct_contextual_retriever_suffix"
  "run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_lateinteraction_contextual_retriever_suffix"
  "run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_multilateinteraction_contextual_retriever_suffix"
)

# Distractor length
distractor_len=20
top_k=10
threshold=0.5
# Create CSV header
echo "Experiment,Mean Average Precision at ${top_k},Mean Reciprocal Rank,Mean NDCG at ${top_k},Mean Precision at ${top_k},Macro-Averaged Precision,Macro-Averaged Recall,Macro-Averaged F1 Score,ROC AUC Score,Mean Average Precision at ${top_k} (Chinese),Mean Reciprocal Rank (Chinese),Mean NDCG at ${top_k} (Chinese),Mean Precision at ${top_k} (Chinese),Macro-Averaged Precision (Chinese),Macro-Averaged Recall (Chinese),Macro-Averaged F1 Score (Chinese),ROC AUC Score (Chinese),Mean Average Precision at ${top_k} (English),Mean Reciprocal Rank (English),Mean NDCG at ${top_k} (English),Mean Precision at ${top_k} (English),Macro-Averaged Precision (English),Macro-Averaged Recall (English),Macro-Averaged F1 Score (English),ROC AUC Score (English)" > context_retrieval_results.csv

# Loop through each folder in the array
for folder in "${exp_folder[@]}"; do
    echo "Processing $folder"

    # Update the exp_path with the current folder
    exp_path="./exp/asr_whisper/${folder}/decode_asr_whisper_ctc_greedy_c${distractor_len}_asr_model_valid.loss.ave_10best/test"

    output=$(python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
      --context_list_path "./local/contextual/rarewords/rareword_f10_test.txt" \
      --ref_context_path "./dump/raw/test/uttblist_idx" \
      --hyp_context_path "${exp_path}/context_text" \
      --hyp_context_prob_path "${exp_path}/context_score" \
      --context_candidate_path "${exp_path}/context_candidate" \
      --k ${top_k} \
      --threshold ${threshold})

    # Function to extract metrics from the output
    extract_metric() {
        local text="$1"
        local pattern="$2"
        echo "$text" | grep -oP "$pattern" | head -n1
    }

    # Overall Metrics
    map10=$(extract_metric "$output" 'Mean Average Precision at [^:]+: \K[0-9.]+')
    mrr=$(extract_metric "$output" 'Mean Reciprocal Rank: \K[0-9.]+')
    ndcg10=$(extract_metric "$output" 'Mean NDCG at [^:]+: \K[0-9.]+')
    precision10=$(extract_metric "$output" 'Mean Precision at [^:]+: \K[0-9.]+')
    precision=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+: \K[0-9.]+')
    recall=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+: \K[0-9.]+')
    f1=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+: \K[0-9.]+')
    roc_auc=$(extract_metric "$output" 'ROC AUC \(Overall\): \K[0-9.]+')

    # Chinese Metrics
    map10_chinese=$(extract_metric "$output" 'Mean Average Precision at [^:]+ \(Chinese\): \K[0-9.]+')
    mrr_chinese=$(extract_metric "$output" 'Mean Reciprocal Rank \(Chinese\): \K[0-9.]+')
    ndcg10_chinese=$(extract_metric "$output" 'Mean NDCG at [^:]+ \(Chinese\): \K[0-9.]+')
    precision10_chinese=$(extract_metric "$output" 'Mean Precision at [^:]+ \(Chinese\): \K[0-9.]+')
    precision_chinese=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    recall_chinese=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    f1_chinese=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    roc_auc_chinese=$(extract_metric "$output" 'ROC AUC \(Chinese\): \K[0-9.]+')

    # English Metrics
    map10_english=$(extract_metric "$output" 'Mean Average Precision at [^:]+ \(English\): \K[0-9.]+')
    mrr_english=$(extract_metric "$output" 'Mean Reciprocal Rank \(English\): \K[0-9.]+')
    ndcg10_english=$(extract_metric "$output" 'Mean NDCG at [^:]+ \(English\): \K[0-9.]+')
    precision10_english=$(extract_metric "$output" 'Mean Precision at [^:]+ \(English\): \K[0-9.]+')
    precision_english=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+ \(English\): \K[0-9.]+')
    recall_english=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+ \(English\): \K[0-9.]+')
    f1_english=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+ \(English\): \K[0-9.]+')
    roc_auc_english=$(extract_metric "$output" 'ROC AUC \(English\): \K[0-9.]+')

    # Ensure that empty variables are set to a placeholder (e.g., "N/A") to prevent CSV misalignment
    map10=${map10:-N/A}
    mrr=${mrr:-N/A}
    ndcg10=${ndcg10:-N/A}
    precision10=${precision10:-N/A}
    precision=${precision:-N/A}
    recall=${recall:-N/A}
    f1=${f1:-N/A}
    roc_auc=${roc_auc:-N/A}

    map10_chinese=${map10_chinese:-N/A}
    mrr_chinese=${mrr_chinese:-N/A}
    ndcg10_chinese=${ndcg10_chinese:-N/A}
    precision10_chinese=${precision10_chinese:-N/A}
    precision_chinese=${precision_chinese:-N/A}
    recall_chinese=${recall_chinese:-N/A}
    f1_chinese=${f1_chinese:-N/A}
    roc_auc_chinese=${roc_auc_chinese:-N/A}

    map10_english=${map10_english:-N/A}
    mrr_english=${mrr_english:-N/A}
    ndcg10_english=${ndcg10_english:-N/A}
    precision10_english=${precision10_english:-N/A}
    precision_english=${precision_english:-N/A}
    recall_english=${recall_english:-N/A}
    f1_english=${f1_english:-N/A}
    roc_auc_english=${roc_auc_english:-N/A}

    # Append the results to the CSV file
    echo "$folder,$map10,$mrr,$ndcg10,$precision10,$precision,$recall,$f1,$roc_auc,$map10_chinese,$mrr_chinese,$ndcg10_chinese,$precision10_chinese,$precision_chinese,$recall_chinese,$f1_chinese,$roc_auc_chinese,$map10_english,$mrr_english,$ndcg10_english,$precision10_english,$precision_english,$recall_english,$f1_english,$roc_auc_english" >> context_retrieval_results.csv

done

echo "Entity pharse"

# Loop through each folder in the array
for folder in "${exp_folder[@]}"; do
    echo "Processing $folder"

    # Update the exp_path with the current folder
    exp_path="./exp/asr_whisper/${folder}/decode_asr_whisper_ctc_greedy_c${distractor_len}_entity_asr_model_valid.loss.ave_10best/test"

    # Run the Python script with the updated exp_path and capture the output
    output=$(python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
      --context_list_path "./local/contextual/rarewords/esun.entity.txt" \
      --ref_context_path "./dump/raw/test/uttblist_idx_entity" \
      --hyp_context_path "${exp_path}/context_text" \
      --hyp_context_prob_path "${exp_path}/context_score" \
      --context_candidate_path "${exp_path}/context_candidate" \
      --k ${top_k} \
      --threshold ${threshold})

    # Function to extract metrics from the output
    extract_metric() {
        local text="$1"
        local pattern="$2"
        echo "$text" | grep -oP "$pattern" | head -n1
    }

    # Overall Metrics
    map10=$(extract_metric "$output" 'Mean Average Precision at [^:]+: \K[0-9.]+')
    mrr=$(extract_metric "$output" 'Mean Reciprocal Rank: \K[0-9.]+')
    ndcg10=$(extract_metric "$output" 'Mean NDCG at [^:]+: \K[0-9.]+')
    precision10=$(extract_metric "$output" 'Mean Precision at [^:]+: \K[0-9.]+')
    precision=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+: \K[0-9.]+')
    recall=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+: \K[0-9.]+')
    f1=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+: \K[0-9.]+')
    roc_auc=$(extract_metric "$output" 'ROC AUC \(Overall\): \K[0-9.]+')

    # Chinese Metrics
    map10_chinese=$(extract_metric "$output" 'Mean Average Precision at [^:]+ \(Chinese\): \K[0-9.]+')
    mrr_chinese=$(extract_metric "$output" 'Mean Reciprocal Rank \(Chinese\): \K[0-9.]+')
    ndcg10_chinese=$(extract_metric "$output" 'Mean NDCG at [^:]+ \(Chinese\): \K[0-9.]+')
    precision10_chinese=$(extract_metric "$output" 'Mean Precision at [^:]+ \(Chinese\): \K[0-9.]+')
    precision_chinese=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    recall_chinese=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    f1_chinese=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+ \(Chinese\): \K[0-9.]+')
    roc_auc_chinese=$(extract_metric "$output" 'ROC AUC \(Chinese\): \K[0-9.]+')

    # English Metrics
    map10_english=$(extract_metric "$output" 'Mean Average Precision at [^:]+ \(English\): \K[0-9.]+')
    mrr_english=$(extract_metric "$output" 'Mean Reciprocal Rank \(English\): \K[0-9.]+')
    ndcg10_english=$(extract_metric "$output" 'Mean NDCG at [^:]+ \(English\): \K[0-9.]+')
    precision10_english=$(extract_metric "$output" 'Mean Precision at [^:]+ \(English\): \K[0-9.]+')
    precision_english=$(extract_metric "$output" 'Macro-Averaged Precision at threshold [^:]+ \(English\): \K[0-9.]+')
    recall_english=$(extract_metric "$output" 'Macro-Averaged Recall at threshold [^:]+ \(English\): \K[0-9.]+')
    f1_english=$(extract_metric "$output" 'Macro-Averaged F1 Score at threshold [^:]+ \(English\): \K[0-9.]+')
    roc_auc_english=$(extract_metric "$output" 'ROC AUC \(English\): \K[0-9.]+')

    # Ensure that empty variables are set to a placeholder (e.g., "N/A") to prevent CSV misalignment
    map10=${map10:-N/A}
    mrr=${mrr:-N/A}
    ndcg10=${ndcg10:-N/A}
    precision10=${precision10:-N/A}
    precision=${precision:-N/A}
    recall=${recall:-N/A}
    f1=${f1:-N/A}
    roc_auc=${roc_auc:-N/A}

    map10_chinese=${map10_chinese:-N/A}
    mrr_chinese=${mrr_chinese:-N/A}
    ndcg10_chinese=${ndcg10_chinese:-N/A}
    precision10_chinese=${precision10_chinese:-N/A}
    precision_chinese=${precision_chinese:-N/A}
    recall_chinese=${recall_chinese:-N/A}
    f1_chinese=${f1_chinese:-N/A}
    roc_auc_chinese=${roc_auc_chinese:-N/A}

    map10_english=${map10_english:-N/A}
    mrr_english=${mrr_english:-N/A}
    ndcg10_english=${ndcg10_english:-N/A}
    precision10_english=${precision10_english:-N/A}
    precision_english=${precision_english:-N/A}
    recall_english=${recall_english:-N/A}
    f1_english=${f1_english:-N/A}
    roc_auc_english=${roc_auc_english:-N/A}

    # Append the results to the CSV file
    echo "$folder,$map10,$mrr,$ndcg10,$precision10,$precision,$recall,$f1,$roc_auc,$map10_chinese,$mrr_chinese,$ndcg10_chinese,$precision10_chinese,$precision_chinese,$recall_chinese,$f1_chinese,$roc_auc_chinese,$map10_english,$mrr_english,$ndcg10_english,$precision10_english,$precision_english,$recall_english,$f1_english,$roc_auc_english" >> context_retrieval_results.csv

done