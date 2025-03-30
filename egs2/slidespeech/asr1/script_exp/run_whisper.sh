#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="S95"
valid_set="dev"
test_sets="test"

asr_config=conf/exp/train_asr_hybrid_whisper_ctc.yaml
inference_config=conf/decode_asr_whisper_hybrid_bs10.yaml
asr_tag=hybird_whisper_ctc

use_lm=false
use_wordlm=false

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
    --inference_asr_model 20epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 "$@"