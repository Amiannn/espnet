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

asr_config=conf/conformer_new/ca.yaml
inference_config=conf/conformer_new/decode/decode_asr_bs3_ctx_keywords.yaml
asr_tag=conformer_new/ca

pretrained_model=../asr1/exp/asr_train_conformer_raw_en_bpe5000_sp_suffix/valid.acc.ave_10best.pth

use_lm=false

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 32 \
    --gpu_inference false \
    --inference_nj 6 \
    --nbpe 5000 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --uttblist_idx_train "${uttblist_idx_train}" \
    --uttblist_idx_valid "${uttblist_idx_valid}" \
    --uttblist_idx_test "${uttblist_idx_test}" \
    --contextualization true \
    --ignore_init_mismatch true \
    --inference_asr_model valid.acc.ave_10best.pth \
    --pretrained_model "${pretrained_model},${pretrained_model}:ctc.ctc_lo.weights:contextualizer.encoder.embed" \
    "$@"