#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="S95"
valid_set="dev"
test_sets="test"

uttblist_idx_train="uttblist_idx_f65536"
uttblist_idx_valid="uttblist_idx_f65536"
uttblist_idx_test="uttblist_idx_keywords"

asr_config=conf/whisper/ca_iw_naive_ca_all.yaml
inference_config=conf/whisper/decode/decode_asr_bs3_ctx_keywords.yaml
asr_tag=whisper/ca_iw_naive_all

use_lm=false
use_wordlm=false

pretrained_model=../asr1/exp/asr_hybird_whisper_ctc/20epoch.pth

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 10 \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --ngpu 1 \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --cleaner whisper_basic \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model 30epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --uttblist_idx_train "${uttblist_idx_train}" \
    --uttblist_idx_valid "${uttblist_idx_valid}" \
    --uttblist_idx_test "${uttblist_idx_test}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --pretrained_model "${pretrained_model}" \
    --contextualization true \
    --lm_fold_length 150 "$@"