#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/exp/train_whisper_base.yaml
inference_config=conf/decode_asr_whisper_noctc_greedy.yaml
asr_tag="whisper_base_finetune_test"

pretrained_model=exp/asr_whisper_base_finetune/3epoch.pth
CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --gpu_inference false \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 10 \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --cleaner whisper_en \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model 3epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --pretrained_model "${pretrained_model}" \
    "$@"