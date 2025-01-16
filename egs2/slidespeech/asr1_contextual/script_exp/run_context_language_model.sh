#!/usr/bin/env bash
# Author: Your Name
# Date:   2025-01-10
# Description:
#   A stand-alone script to collect LM statistics, train the language model,
#   and calculate perplexity in ESPnet2.

# Exit on error, unset variables are errors, and pipelines fail on the first error
set -e
set -u
set -o pipefail

# ----------------
# Configuration
# ----------------

. ./path.sh
. ./cmd.sh

# Commands
# train_cmd="run.pl"         # or queue.pl / slurm.pl depending on your cluster
# cuda_cmd="run.pl"          # Command for CUDA jobs; adjust if using a scheduler like slurm
python="python3"           # Path to your Python interpreter

# Language Model Configuration
bpemodel="./data/en_token_list/bpe_unigram5000suffix/bpe.model"
lm_token_type="bpe"
lm_token_list="./data/en_token_list/bpe_unigram5000suffix/tokens.txt"

lm_config="./conf/train_lm_transformer2.yaml"

# Paths to training, development, and test text
lm_train_text="./dump/raw/S95_sp/uttblist_f65536_merged"
lm_dev_text="./dump/raw/dev/uttblist_f65536_merged"
lm_test_text="./dump/raw/S95_sp/uttblist_f65536_merged"  # Added for perplexity calculation

# Number of parallel jobs
nj=2

# Directories
lm_stats_dir="./exp/lm/source_domain"
logdir="${lm_stats_dir}/logdir"

# Optional settings
nlsyms_txt=""            # Non-linguistic symbols file (set empty if not used)
cleaner="none"           # Text cleaning type
g2p="none"               # Grapheme-to-phoneme conversion type (if required)

# Stage 7 Specific Settings
num_splits_lm=1          # Number of splits for LM training (set >1 if needed)
lm_fold_length=150       # Fold length for batching; adjust based on your data and GPU memory
inference_lm="1000epoch.pth"  # Trained LM model file name

# Training Settings
ngpu=1                   # Number of GPUs
num_nodes=1              # Number of nodes

# Run Arguments (if any additional args are needed, specify here)
run_args=""              # e.g., "--resume true"

# ----------------
# 1. Prepare logdir & parallel splits
# ----------------

echo "===== Preparing log directories and splitting data ====="
mkdir -p "${logdir}"
mkdir -p "${lm_stats_dir}/splits${num_splits_lm}"
mkdir -p "${lm_stats_dir}/perplexity_test"

# Determine the minimum number of parallel jobs based on data size
num_train_lines=$(wc -l < "${lm_train_text}")
num_dev_lines=$(wc -l < "${lm_dev_text}")
min_nj=$(awk -v a="${nj}" -v b="${num_train_lines}" -v c="${num_dev_lines}" \
             'BEGIN{print (a < b ? a : b) < c ? (a < b ? a : b) : c}')
_actual_nj=$min_nj

echo "Using ${_actual_nj} parallel jobs."

# Split training text
split_train_scps=""
for n in $(seq "${_actual_nj}"); do
    split_train_scps+=" ${logdir}/train.${n}.scp"
done
utils/split_scp.pl "${lm_train_text}" ${split_train_scps}

# Split dev text
split_dev_scps=""
for n in $(seq "${_actual_nj}"); do
    split_dev_scps+=" ${logdir}/dev.${n}.scp"
done
utils/split_scp.pl "${lm_dev_text}" ${split_dev_scps}

# ----------------
# 2. LM Collect-Stats (Stage 6)
# ----------------

echo "===== Stage 6: LM Statistics Collection ====="
mkdir -p "${lm_stats_dir}"

# Run parallel jobs for stats collection
${train_cmd} JOB=1:"${_actual_nj}" "${logdir}"/stats.JOB.log \
    ${python} -m espnet2.bin.lm_train \
        --collect_stats true \
        --use_preprocessor true \
        --bpemodel "${bpemodel}" \
        --token_type "${lm_token_type}" \
        --token_list "${lm_token_list}" \
        ${nlsyms_txt:+--non_linguistic_symbols "${nlsyms_txt}"} \
        --cleaner "${cleaner}" \
        --g2p "${g2p}" \
        --train_data_path_and_name_and_type "${lm_train_text},text,text" \
        --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
        --train_shape_file "${logdir}/train.JOB.scp" \
        --valid_shape_file "${logdir}/dev.JOB.scp" \
        --output_dir "${logdir}/stats.JOB" \
        ${lm_config:+--config "${lm_config}"} || { echo "Error in LM collect-stats"; exit 1; }

# ----------------
# 3. Aggregate Stats
# ----------------

echo "===== Aggregating LM Statistics ====="
opts=""
for i in $(seq "${_actual_nj}"); do
    opts+="--input_dir ${logdir}/stats.${i} "
done

${python} -m espnet2.bin.aggregate_stats_dirs \
    ${opts} \
    --output_dir "${lm_stats_dir}"

# Append the num-tokens at the last dimension (for batch-bin counts)
num_tokens=$(wc -l < "${lm_token_list}")
awk -v N="$num_tokens" '{ print $0 "," N }' \
    < "${lm_stats_dir}/train/text_shape" \
    > "${lm_stats_dir}/train/text_shape.${lm_token_type}"

awk -v N="$num_tokens" '{ print $0 "," N }' \
    < "${lm_stats_dir}/valid/text_shape" \
    > "${lm_stats_dir}/valid/text_shape.${lm_token_type}"

echo "===== Stage 6 Completed: LM Statistics Collected ====="

# ----------------
# 4. LM Training (Stage 7)
# ----------------

echo "===== Stage 7: LM Training ====="

_opts=""
if [ -n "${lm_config}" ]; then
    _opts+="--config ${lm_config} "
fi

if [ "${num_splits_lm}" -gt 1 ]; then
    # Split the training data to limit memory usage
    split_dir="${lm_stats_dir}/splits${num_splits_lm}"
    if [ ! -f "${split_dir}/.done" ]; then
        echo "Splitting LM training data into ${num_splits_lm} parts..."
        rm -f "${split_dir}/.done"
        ${python} -m espnet2.bin.split_scps \
            --scps "${lm_train_text}" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
            --num_splits "${num_splits_lm}" \
            --output_dir "${split_dir}"
        touch "${split_dir}/.done"
    else
        echo "Split directory '${split_dir}/.done' exists. Skipping splitting."
    fi

    _opts+="--train_data_path_and_name_and_type ${split_dir}/lm_train.txt,text,text "
    _opts+="--train_shape_file ${split_dir}/text_shape.${lm_token_type} "
    _opts+="--multiple_iterator true "
else
    _opts+="--train_data_path_and_name_and_type ${lm_train_text},text,text "
    _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
fi

# Launch LM training using distributed training if needed
echo "Launching LM training..."
if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
    # SGE can't include "/" in a job name
    jobname="$(basename ${lm_stats_dir})"
else
    jobname="${lm_stats_dir}/train.log"
fi

${python} -m espnet2.bin.launch \
    --cmd "${cuda_cmd} --name ${jobname}" \
    --log "${lm_stats_dir}/train.log" \
    --ngpu "${ngpu}" \
    --num_nodes "${num_nodes}" \
    --init_file_prefix "${lm_stats_dir}/.dist_init_" \
    --multiprocessing_distributed true -- \
    ${python} -m espnet2.bin.lm_train \
        --ngpu "${ngpu}" \
        --use_preprocessor true \
        --bpemodel "${bpemodel}" \
        --token_type "${lm_token_type}" \
        --token_list "${lm_token_list}" \
        ${nlsyms_txt:+--non_linguistic_symbols "${nlsyms_txt}"} \
        --cleaner "${cleaner}" \
        --g2p "${g2p}" \
        --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
        --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
        --fold_length "${lm_fold_length}" \
        --resume true \
        --output_dir "${lm_stats_dir}" \
        ${_opts} ${run_args} || { echo "Error in LM training"; exit 1; }

# echo "===== Stage 7 Completed: LM Training Finished ====="

# ----------------
# 5. Perplexity Calculation (Stage 8)
# ----------------

echo "===== Stage 8: Perplexity Calculation ====="
mkdir -p "${lm_stats_dir}/perplexity_test"

# Calculate perplexity
echo "Calculating perplexity on test set..."
${cuda_cmd} --gpu "${ngpu}" "${lm_stats_dir}/perplexity_test/lm_calc_perplexity.log" \
    ${python} -m espnet2.bin.lm_calc_perplexity \
        --ngpu "${ngpu}" \
        --data_path_and_name_and_type "${lm_test_text},text,text" \
        --train_config "${lm_stats_dir}/config.yaml" \
        --model_file "${lm_stats_dir}/${inference_lm}" \
        --output_dir "${lm_stats_dir}/perplexity_test" \
        || { echo "Error in perplexity calculation"; exit 1; }

# Display perplexity result
if [ -f "${lm_stats_dir}/perplexity_test/ppl" ]; then
    ppl=$(cat "${lm_stats_dir}/perplexity_test/ppl")
    echo "===== Perplexity ====="
    echo "PPL for '${lm_test_text}': ${ppl}"
else
    echo "Perplexity file not found. Please check the logs."
fi

echo "===== Stage 8 Completed: Perplexity Calculation Finished ====="
echo "===== All Stages Completed Successfully ====="
echo "Check '${lm_stats_dir}' for all outputs and logs."
