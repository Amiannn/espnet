#!/usr/bin/env bash
set -e

# Define your stage variables
stage=11
stop_stage=11

# Define an array of the script base paths (without options)
commands=(
  "./script_exp/conformer_new/run_ca_enc.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_naive_all.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_bbse_all.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_explicit_all.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_naive_ca.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_bbse_ca.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_explicit_ca.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_naive_out.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_bbse_out.sh"
  "./script_exp/conformer_new/run_ca_enc_iw_explicit_out.sh"
)

# Get the total number of commands.
total_commands=${#commands[@]}

# Iterate over each command and display progress.
for index in "${!commands[@]}"; do
  current=$(( index + 1 ))
  
  # Build the full command with the options.
  # You can include as many parameters as you need.
  full_command="${commands[$index]} --stage ${stage} --stop_stage ${stop_stage}"
  
  echo "[$current/$total_commands] Running: ${full_command}"
  
  # Execute the command using eval.
  eval "${full_command}"
  
  echo "Finished command $current/$total_commands"
  echo "----------------------------------------"
done

echo "All commands finished."
