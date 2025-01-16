ref_path="./dump/raw/test/text"
context_type="keywords"
context_path="./local/contextual/metadata/related_files/test/${context_type}_fix"
wav2session="./dump/raw/test/wav2session"
hyp_path="./exp/asr_conformer/run_context_adapter_encoder_reweight0.8_suffix/decode_asr_no_contextual_bs3_asr_model_valid.acc.ave_10best/test/text"

# Determine the directory of hyp_path
output_dir=$(dirname "$hyp_path")
output_file="$output_dir/results_${context_type}"

# Run the script with correct arguments and redirect output
python3 -m local.test.compute_wer_details --verbose 1 \
    --ref "$ref_path" \
    --ref_ocr "$context_path" \
    --rec_name base \
    --ref2session "$wav2session" \
    --rec_file "$hyp_path" \
    > "$output_file"
