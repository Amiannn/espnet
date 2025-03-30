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
inference_config=conf/decode_asr_whisper.yaml
asr_tag=hybird_whisper_ctc_bpe

use_lm=false
use_wordlm=false

bpe_nlsyms=""
nbpe=5000

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 10 \
    --gpu_inference false \
    --inference_nj 10 \
    --ngpu 1 \
    --nbpe ${nbpe} \
    --suffixbpe suffix \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --use_word_lm ${use_wordlm}   \
    --feats_normalize '' \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model 64epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --score_opts "-e utf-8 -c NOASCII" \
    "$@"
