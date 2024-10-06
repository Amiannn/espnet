# distractor length
distractor_len=100
# Define variables
inference_config="conf/contextual/whisper/decode_asr_whisper_ctc_greedy_c${distractor_len}.yaml"
testset="test"
uttblist_idx_test="uttblist_idx"
nj=20

# Array of script names
scripts=(
  # "run_medium_dotproduct_contextual_retriever_balanced_alpha0.8"
  # "run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8"
  # "run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8"
  # "run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8"
  # "run_medium_dotproduct_contextual_retriever"
  # "run_medium_xdotproduct_contextual_retriever"
  # "run_medium_lateinteraction_contextual_retriever"
  "run_medium_multilateinteraction_contextual_retriever"
)

# Loop through each script and run it
for script in "${scripts[@]}"; do
  ./scripts_exp/whisper/${script}.sh \
    --test_sets ${testset} \
    --uttblist_idx_test ${uttblist_idx_test} \
    --inference_config ${inference_config} \
    --inference_nj ${nj} \
    --stage 12
done

# decode using entity

scripts=(
  "run_medium_dotproduct_contextual_retriever_balanced_alpha0.8"
  "run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8"
  "run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8"
  "run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8"
  "run_medium_dotproduct_contextual_retriever"
  "run_medium_xdotproduct_contextual_retriever"
  "run_medium_lateinteraction_contextual_retriever"
  "run_medium_multilateinteraction_contextual_retriever"
)

inference_config="conf/contextual/whisper/decode_asr_whisper_ctc_greedy_c${distractor_len}_entity.yaml"
uttblist_idx_test="uttblist_idx_entity"

# Loop through each script and run it
# for script in "${scripts[@]}"; do
#   ./scripts_exp/whisper/${script}.sh \
#     --test_sets ${testset} \
#     --uttblist_idx_test ${uttblist_idx_test} \
#     --inference_config ${inference_config} \
#     --inference_nj ${nj} \
#     --stage 12
# done