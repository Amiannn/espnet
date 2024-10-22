#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=13

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/whisper/train_asr_whisper_freeze_ctc.yaml
inference_config=conf/whisper/decode_asr_whisper_noctc_greedy.yaml
asr_tag=whisper_medium_ctc_finetune_bpe8000

lm_config=conf/exp/train_lm_transformer.yaml
use_lm=false
use_wordlm=false
nbpe=8000

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 20 \
    --gpu_inference false \
    --inference_nj 1 \
    --ngpu 1 \
    --lang zh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nbpe ${nbpe} \
    --suffixbpe suffix \
    --feats_normalize '' \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model 2epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/corpus.txt" \
    --score_opts "-e utf-8 -c NOASCII" \
    "$@"