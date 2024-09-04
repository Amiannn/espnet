#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

uttblist_idx_train="uttblist_idx_f1000"
uttblist_idx_valid="uttblist_idx_f1000"
uttblist_idx_test="uttblist_idx_entity"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_lora_prefix_tuning.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper_prefix_tuning_entity.yaml
asr_tag=whisper/run_medium_prefix_tuning

# pretrained_model=../asr1/exp/asr_whisper_medium_lora_decoder/3epoch.pth
CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 5 \
    --gpu_inference true \
    --inference_nj 2 \
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
    --uttblist_idx_train "${uttblist_idx_train}" \
    --uttblist_idx_valid "${uttblist_idx_valid}" \
    --uttblist_idx_test "${uttblist_idx_test}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --inference_asr_model 3epoch.pth \
    --contextualization true \
    --lm_fold_length 150 "$@"

    # --pretrained_model "${pretrained_model}" \

