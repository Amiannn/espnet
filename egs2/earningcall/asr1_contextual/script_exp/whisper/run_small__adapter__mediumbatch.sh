#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/contextual_adapter/whisper/tune_small_adapter.yaml
inference_config=conf/exp/decode_contextual_whisper_xphone_greedy.yaml
asr_tag=whisper/tune_small_adapter__bpe5000

pretrained_model=../asr1/exp/asr_whisper_small_finetune_lr1e-5_adamw_wd1e-2_3epochs_tinybatch_suffix/valid.acc.ave_3best.pth
CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 2 \
    --gpu_inference false \
    --inference_nj 5 \
    --nbpe 5000 \
    --suffixbpe suffix \
    --feats_normalize '' \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.acc.ave_3best.pth \
    --contextualization true \
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.decoders.token_embedding:contextualizer.encoder.embed" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"