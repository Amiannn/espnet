#!/usr/bin/env bash
set -e

# Base directory where all experiment folders live
BASE_DIR="./exp/asr_conformer_new"

echo "Searching for experiment folders under: $BASE_DIR"
echo

# Loop through each experiment folder under BASE_DIR
for exp_folder in "$BASE_DIR"/*; do
  if [ -d "$exp_folder" ]; then
    echo "Processing experiment folder: $exp_folder"

    # Within this experiment folder, look for subfolders whose names start with "decode_"
    for decode_dir in "$exp_folder"/decode_*; do
      if [ -d "$decode_dir" ]; then
        echo "  Found decode folder: $decode_dir"

        # Construct the hyp_path by appending "/test/text" to the decode folder path
        hyp_path="${decode_dir}/dev/text"
        echo "  Generated hyp_path: $hyp_path"
        
        # Check if the hypothesis file exists. 
        # Adjust the test below depending on whether hyp_path is a file or a directory.
        # In your case, it seems to be a file that should exist.
        if [ ! -f "$hyp_path" ]; then
          echo "Warning: Hypothesis file not found at '$hyp_path'. Skipping run for this decode folder."
          continue
        fi
        
        echo "----------------------------------------"
        echo "Running run.sh with hyp_path: $hyp_path"
        echo "----------------------------------------"
        
        # Run run.sh with the generated hyp_path by overriding the hyp_path variable.
        # (This assumes that run.sh checks for the OVERRIDE_HYP_PATH variable, as shown below.)
        OVERRIDE_HYP_PATH="$hyp_path" ./script_exp/scoring/run.sh

        echo "Finished running run.sh for decode folder: $decode_dir"
        echo
      fi
    done
    echo "Done processing experiment folder: $exp_folder"
    echo "========================================"
    echo
  fi
done

python3 local/test/aggregate_csv.py --root_dir $BASE_DIR
echo "All experiments finished."
