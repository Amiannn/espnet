#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_softprompt.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper.yaml
asr_tag=whisper/run_medium_softprompt


CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 15 \
    --gpu_inference true \
    --inference_nj 4 \
    --lang zh \
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
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --inference_asr_model 0epoch.pth \
    --contextualization true \
    --lm_fold_length 150 "$@"

