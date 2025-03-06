#!/bin/bash

# Set the target directory for scripts and the log directory
SCRIPT_DIR="./script_exp/conformer_new"
LOG_DIR="./exp/decode_exp"
valid_set="test"
test_sets="dev"
model_name="valid.acc.ave_10best"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Define arrays of inference configurations and uttblist index tests
inference_configs=( "conf/conformer_new/decode/decode_asr_bs20_ctx_keywords_dev.yaml" )
uttblist_idx_tests=( "uttblist_idx_keywords" )

# Ensure both arrays have the same length
if [[ ${#inference_configs[@]} -ne ${#uttblist_idx_tests[@]} ]]; then
    echo "Error: The number of inference configurations and uttblist indices must match."
    exit 1
fi

# Define a skip list for scripts (without the .sh extension)
skip_list=( "run_ca_enc_iw_naive_all" "run_ca_enc_iw_bbse_all" "run_ca_enc_iw_explicit_all" "run_ca_enc" )

# Determine the total number of script runs for progress tracking.
# Total runs = (# of inference configs) * (# of executable scripts in SCRIPT_DIR)
script_count=0
for script in "$SCRIPT_DIR"/*.sh; do
    if [[ -x "$script" ]]; then
        # Check if the script is in the skip list
        script_base=$(basename "$script" .sh)
        skip=false
        for skip_script in "${skip_list[@]}"; do
            if [[ "$script_base" == "$skip_script" ]]; then
                skip=true
                break
            fi
        done
        if $skip; then
            continue
        fi

        ((script_count++))
    fi
done

if (( script_count == 0 )); then
    echo "No executable (and non-skipped) scripts found in $SCRIPT_DIR."
    exit 1
fi

total_runs=$(( ${#inference_configs[@]} * script_count ))
current_run=0

# Iterate through each inference configuration and uttblist index test pair
for (( i=0; i<${#inference_configs[@]}; i++ )); do
    inference_config="${inference_configs[i]}"
    uttblist_idx_test="${uttblist_idx_tests[i]}"

    echo "========================================"
    echo "Processing pair $((i+1)) of ${#inference_configs[@]}"
    echo "Inference Config: $inference_config"
    echo "Uttblist Index Test: $uttblist_idx_test"
    echo "========================================"

    # Execute each executable script in the SCRIPT_DIR
    for script in "$SCRIPT_DIR"/*.sh; do
        if [[ -x "$script" ]]; then
            # Check if the script should be skipped based on the skip list
            script_base=$(basename "$script" .sh)
            skip=false
            for skip_script in "${skip_list[@]}"; do
                if [[ "$script_base" == "$skip_script" ]]; then
                    echo "Skipping: $script_base (matches skip list)"
                    skip=true
                    break
                fi
            done
            if $skip; then
                continue
            fi

            ((current_run++))
            echo "[$current_run / $total_runs] Running: $(basename "$script") with --inference_config $inference_config --uttblist_idx_test $uttblist_idx_test"

            # Construct a log file name based on script and config names
            config_base=$(basename "$inference_config" .yaml)
            log_file="$LOG_DIR/${script_base}_${config_base}_${uttblist_idx_test}.log"
            echo "Logging output to: $log_file"

            # Run the script with the parameters and log the output (both stdout and stderr)
            "$script" \
                --stage 12 \
                --stop_stage 20 \
                --inference_nj 12 \
                --inference_asr_model "${model_name}.pth" \
                --inference_config "$inference_config" \
                --uttblist_idx_test "$uttblist_idx_test" \
                --valid_set "${valid_set}" \
                --test_sets "${test_sets}" \
                "$@" 2>&1 | tee "$log_file"

            echo "Finished run $current_run of $total_runs."
            echo "----------------------------------------"
        else
            echo "Skipping: $script (not executable)"
        fi
    done
done
