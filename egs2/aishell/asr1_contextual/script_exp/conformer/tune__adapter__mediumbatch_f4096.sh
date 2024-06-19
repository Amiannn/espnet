#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="test"

asr_config=conf/contextual/conformer/adapter__mediumbatch_f4096.yaml
inference_config=conf/contextual/conformer/decode_asr_conformer_adapter_bs5.yaml
asr_tag=conformer/adapter__mediumbatch_f4096

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

pretrained_model=../asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth
CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 32 \
    --inference_nj 10 \
    --ngpu 1 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_tag "${asr_tag}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --inference_asr_model valid.acc.ave_10best.pth \
    --contextualization true \
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.embed.0:contextualizer.encoder.embed" \
    "$@"