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

uttblist_idx_train="uttblist_idx_f65536"
uttblist_idx_valid="uttblist_idx_f65536"
uttblist_idx_test="uttblist_idx_f10"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_dotproduct_contextual_retriever.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper_ctc_greedy_c100.yaml
asr_tag=whisper/run_medium_dotproduct_contextual_retriever


lm_config=conf/exp/train_lm_transformer.yaml
use_lm=false
use_wordlm=false
nbpe=8000

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 20 \
    --lang zh\
    --gpu_inference false \
    --inference_nj 20 \
    --ngpu 1 \
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
    --inference_asr_model valid.loss.ave_10best.pth \
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
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/corpus.txt" \
    --score_opts "-e utf-8 -c NOASCII" \
    "$@"

    # --pretrained_model "${pretrained_model},${pretrained_model}:ctc.ctc_lo.weights:contextualizer.encoder.embed" \
