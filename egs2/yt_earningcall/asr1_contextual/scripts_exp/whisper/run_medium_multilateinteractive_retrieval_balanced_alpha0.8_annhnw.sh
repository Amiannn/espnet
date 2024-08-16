#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_multilateinteractive_retrieval_balanced_alpha0.8_annhnw.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper_ctc_greedy_c20_k5.yaml
asr_tag=whisper/run_medium_multilateinteractive_retrieval_balanced_alpha0.8_annhnw

pretrained_model=/home/ubuntu/espnet/egs2/esun/asr1/exp/asr_whisper_medium/0epoch.pth
CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 1 \
    --gpu_inference true \
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
    --inference_asr_model valid.acc.ave_3best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --contextualization true \
    --ignore_init_mismatch true \
    --pretrained_model "${pretrained_model}:decoder.decoders.token_embedding:ctc.ctc_lo,${pretrained_model}:decoder.decoders.token_embedding:contextualizer.encoder.embed" \
    "$@"