# Define the array of experiment folder names
exp_folder=(
  "run_medium_dotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_dotproduct_contextual_retriever_suffix"
  "run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_xdotproduct_contextual_retriever_suffix"
  "run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_lateinteraction_contextual_retriever_suffix"
  "run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
)

# Create CSV header
echo "Experiment,Mean Average Precision at 10,Mean Reciprocal Rank,Mean NDCG at 10,Mean Precision at 10,Mean Precision,Mean Recall,Mean F1 Score,ROC AUC Score" > context_retrieval_results.csv

# Loop through each folder in the array
for folder in "${exp_folder[@]}"; do
    echo "Processing $folder"

    # Update the exp_path with the current folder
    exp_path="./exp/asr_whisper/${folder}/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test"

    # Run the Python script with the updated exp_path and capture the output
    output=$(python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
      --context_list_path "./local/contextual/rarewords/rareword_f10_test.txt" \
      --ref_context_path "./dump/raw/test/uttblist" \
      --hyp_context_path "${exp_path}/context_text" \
      --hyp_context_prob_path "${exp_path}/context_score" \
      --context_candidate_path "${exp_path}/context_candidate" \
      --k 10)
    
    # Extract the values using grep and awk (or sed) and format them into CSV format
    map10=$(echo "$output" | grep "Mean Average Precision at 10:" | awk '{print $NF}')
    mrr=$(echo "$output" | grep "Mean Reciprocal Rank:" | awk '{print $NF}')
    ndcg10=$(echo "$output" | grep "Mean NDCG at 10:" | awk '{print $NF}')
    precision10=$(echo "$output" | grep "Mean Precision at 10:" | awk '{print $NF}')
    precision=$(echo "$output" | grep "^Mean Precision:" | awk '{print $NF}')
    recall=$(echo "$output" | grep "Mean Recall:" | awk '{print $NF}')
    f1=$(echo "$output" | grep "Mean F1 Score:" | awk '{print $NF}')
    roc_auc=$(echo "$output" | grep "ROC AUC Score:" | awk '{print $NF}')

    # Append the results to the CSV file
    echo "$folder,$map10,$mrr,$ndcg10,$precision10,$precision,$recall,$f1,$roc_auc" >> context_retrieval_results.csv
done

echo "Entity pharse"

# Loop through each folder in the array
for folder in "${exp_folder[@]}"; do
    echo "Processing $folder"

    # Update the exp_path with the current folder
    exp_path="./exp/asr_whisper/${folder}/decode_asr_whisper_ctc_greedy_c100_entity_asr_model_valid.loss.ave_10best/test"

    # Run the Python script with the updated exp_path and capture the output
    output=$(python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
      --context_list_path "./local/contextual/rarewords/rareword_f10_test.txt" \
      --ref_context_path "./dump/raw/test/uttblist_entity" \
      --hyp_context_path "${exp_path}/context_text" \
      --hyp_context_prob_path "${exp_path}/context_score" \
      --context_candidate_path "${exp_path}/context_candidate" \
      --k 10)
    
    # Extract the values using grep and awk (or sed) and format them into CSV format
    map10=$(echo "$output" | grep "Mean Average Precision at 10:" | awk '{print $NF}')
    mrr=$(echo "$output" | grep "Mean Reciprocal Rank:" | awk '{print $NF}')
    ndcg10=$(echo "$output" | grep "Mean NDCG at 10:" | awk '{print $NF}')
    precision10=$(echo "$output" | grep "Mean Precision at 10:" | awk '{print $NF}')
    precision=$(echo "$output" | grep "^Mean Precision:" | awk '{print $NF}')
    recall=$(echo "$output" | grep "Mean Recall:" | awk '{print $NF}')
    f1=$(echo "$output" | grep "Mean F1 Score:" | awk '{print $NF}')
    roc_auc=$(echo "$output" | grep "ROC AUC Score:" | awk '{print $NF}')

    # Append the results to the CSV file
    echo "$folder,$map10,$mrr,$ndcg10,$precision10,$precision,$recall,$f1,$roc_auc" >> context_retrieval_results.csv
done