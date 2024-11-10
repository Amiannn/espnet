# Define variables
nj=20
testset="test"

# decode using entity
scripts=(
  "run_medium_contextual_adapter_decoder_pretrained"
)
uttblist_idx_test="uttblist_idx_entity_earningcall"

for distractor_len in 20 100 300; do

  inference_config="conf/contextual/whisper/decode_asr_whisper_contextual_adapter_decoder_c${distractor_len}_entity_earningcall.yaml"
  # Loop through each script and run it
  for script in "${scripts[@]}"; do
    ./scripts_exp/whisper/${script}.sh \
      --test_sets ${testset} \
      --uttblist_idx_test ${uttblist_idx_test} \
      --inference_config ${inference_config} \
      --inference_nj ${nj} \
      --stage 12
  done
done

scripts=(
  "run_medium_contextual_adapter_decoder_pretrained"
)
uttblist_idx_test="uttblist_idx"

for distractor_len in 20 100 1000; do

  inference_config="conf/contextual/whisper/decode_asr_whisper_contextual_adapter_decoder_c${distractor_len}.yaml"
  # Loop through each script and run it
  for script in "${scripts[@]}"; do
    ./scripts_exp/whisper/${script}.sh \
      --test_sets ${testset} \
      --uttblist_idx_test ${uttblist_idx_test} \
      --inference_config ${inference_config} \
      --inference_nj ${nj} \
      --stage 12
  done
done

