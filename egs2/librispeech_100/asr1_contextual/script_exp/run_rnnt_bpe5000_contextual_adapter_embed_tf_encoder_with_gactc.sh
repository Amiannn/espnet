#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/exp/contextual_adapter/train_rnnt_contextual_adapter_embed_tf_encoder_with_gactc.yaml
inference_config=conf/exp/decode_asr_contextual_rnnt_transducer_greedy.yaml
asr_tag=finetune_freeze_ct_bpe5000_enc_cb_embed_tf_gactc

pretrained_model=exp/asr_train_rnnt_conformer_ngpu4_raw_en_bpe5000_sp/valid.loss.ave_5best.pth

CUDA_VISIBLE_DEVICES=1 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 10 \
    --nbpe 5000 \
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
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.embed:contextualizer.encoder.embed" \
    --inference_asr_model valid.loss.best.pth \
    --asr_tag ${asr_tag} \
    "$@"

    # --asr_args "--use_wandb true --wandb_project Contextualize_RNNT" \
    # --ignore_init_mismatch true \
    # --pretrained_model "${pretrained_model}" \
