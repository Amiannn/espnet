#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/contextual_adapter/whisper/train_whisper_tiny_en_contextual_adapter_encoder.yaml
inference_config=conf/exp/decode_contextual_whisper_greedy.yaml

pretrained_model=exp/asr_train_whisper_tiny_en_raw_en_whisper_en_sp/3epoch.pth

CUDA_VISIBLE_DEVICES=1 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 5 \
    --token_type whisper_en \
    --feats_normalize "" \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --cleaner whisper_en \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --contextualization true \
    --pretrained_model $pretrained_model \
    --ignore_init_mismatch true \
    --inference_asr_model valid.acc.best.pth \
    "$@"

    # --asr_args "--use_wandb true --wandb_project Contextualize_Whisper" \

