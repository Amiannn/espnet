# Define the array of experiment folder names
exp_folder=(
  "run_medium_dotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_dotproduct_contextual_retriever_suffix"
  "run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_lateinteraction_contextual_retriever_suffix"
  "run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix"
  "run_medium_xdotproduct_contextual_retriever_suffix"
)

# Loop through each folder in the array
for folder in "${exp_folder[@]}"; do
    echo $folder

  # Update the exp_path with the current folder
  exp_path="./exp/asr_whisper/${folder}/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test"

  # Run the Python script with the updated exp_path
  python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
    --context_list_path "./local/contextual/rarewords/rareword_f10_test.txt" \
    --ref_context_path "./dump/raw/test/uttblist" \
    --hyp_context_path "${exp_path}/context_text" \
    --hyp_context_prob_path "${exp_path}/context_score" \
    --context_candidate_path "${exp_path}/context_candidate" \
    --k 5
done
