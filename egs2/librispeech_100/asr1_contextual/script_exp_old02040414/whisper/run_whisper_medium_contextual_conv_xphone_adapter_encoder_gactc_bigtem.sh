#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/contextual_adapter/whisper/train_whisper_medium_contextual_conv_xphone_adapter_encoder_gactc_bigtem.yaml
inference_config=conf/exp/decode_contextual_whisper_xphone_greedy.yaml
asr_tag=finetune_freeze_whisper_medium_cb_conv_xphone_gactc_bigtem

pretrained_model=/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1/exp/asr_whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs/valid.acc.ave.pth

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --gpu_inference false \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 2 \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --cleaner whisper_en \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --contextualization true \
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.decoders.token_embedding:contextualizer.encoder.embed" \
    --inference_asr_model valid.loss.ave_10best.pth \
    --ignore_init_mismatch true \
    "$@"

    # --asr_args "--use_wandb true --wandb_project Contextualize_Whisper" \
