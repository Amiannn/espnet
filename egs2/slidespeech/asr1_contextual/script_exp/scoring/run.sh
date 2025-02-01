#!/usr/bin/env bash
set -e

# 1) Basic paths (modify as needed)
ref_path="./dump/raw/test/text"
hyp_path="exp/asr_conformer/run_context_adapter_encoder_iw_all_explicit_drop_suffix/decode_asr_no_contextual_bs3_asr_model_valid.acc.ave_10best/test/text"

# 2) If you want to loop over multiple session files, list them here:
#    Otherwise, you can just define one session file.
session_files=(
#   "./dump/raw/test/wav2session"
  "./dump/raw/test/wav2session_domains"
)

# 3) Context types to test:
context_types=(
  "ocr"
  "keywords"
#   "f10"
)

# 4) Loop over each context + session file
for context_type in "${context_types[@]}"; do

  # Decide how to build 'context_path' based on context_type
  # Adjust these paths to your actual directories
  if [ "$context_type" == "ocr" ]; then
    context_path="./local/contextual/metadata/related_files/test/ocr_fix"
  elif [ "$context_type" == "keywords" ]; then
    context_path="./local/contextual/metadata/related_files/test/keywords_fix"
  else
    # e.g. "f10"
    context_path="./dump/raw/test/uttblist_f10"
  fi

  for session_file in "${session_files[@]}"; do
    # Extract the session short name from the file path, e.g. "wav2session" or "wav2session_domains"
    session_name="$(basename "$session_file")"

    # Build output filename based on the context type and session name
    output_dir="$(dirname "$hyp_path")"
    output_file="${output_dir}/results_${context_type}_${session_name}"

    echo "Running context='${context_type}', session='${session_name}'"
    echo "   ref_path      = ${ref_path}"
    echo "   context_path  = ${context_path}"
    echo "   wav2session   = ${session_file}"
    echo "   hyp_path      = ${hyp_path}"
    echo "   output_file   = ${output_file}"
    echo

    # 5) Run the Python script
    python3 -m local.test.compute_wer_details_refactor --verbose 1 \
      --ref "${ref_path}" \
      --ref_ocr "${context_path}" \
      --rec_name base \
      --ref2session "${session_file}" \
      --rec_file "${hyp_path}" \
      > "${output_file}"

  done
done

echo "All done!"
