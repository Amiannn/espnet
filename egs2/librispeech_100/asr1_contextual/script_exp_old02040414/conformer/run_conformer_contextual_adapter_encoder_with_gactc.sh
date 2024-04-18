#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/contextual_adapter/conformer/train_conformer_contextual_adapter_encoder_with_gactc.yaml
inference_config=conf/exp/decode_asr_greedy.yaml
asr_tag=finetune_freeze_con_enc_cb_gactc

pretrained_model=/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1/exp/asr_train_conformer_raw_en_bpe600_sp_suffix/valid.acc.best.pth

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 10 \
    --nbpe 600 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --contextualization true \
    --inference_asr_model valid.acc.best.pth \
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.embed.0:contextualizer.encoder.embed" \
    --asr_tag ${asr_tag} \
    "$@"

    # --ignore_init_mismatch true \
    # --asr_args "--use_wandb true --wandb_project Contextualize_ASR_NEW" \
