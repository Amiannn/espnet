#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

uttblist_idx_train="uttblist_idx_f65536.txt"
uttblist_idx_valid="uttblist_idx_f65536.txt"
uttblist_idx_test="uttblist_idx"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_contextual_adapter_decoder_reweight.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper_contextual_adapter_decoder_c1000_test.yaml
asr_tag=whisper/run_medium_contextual_adapter_decoder_reweight

pretrained_model=exp/asr_whisper/run_medium_contextual_adapter_decoder_pretrained_ce_alpha0.9/valid.loss.ave_10best_fixed.pth

use_lm=false

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 20 \
    --gpu_inference false \
    --inference_nj 1 \
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
    --inference_asr_model valid.loss.ave_10best_fixed.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --uttblist_idx_train "${uttblist_idx_train}" \
    --uttblist_idx_valid "${uttblist_idx_valid}" \
    --uttblist_idx_test "${uttblist_idx_test}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --contextualization true \
    --lm_fold_length 150 \
    --pretrained_model "${pretrained_model}" \
    --ignore_init_mismatch true \
    "$@"

