#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

uttblist_idx_train="uttblist_idx_f1000.txt"
uttblist_idx_valid="uttblist_idx_f1000.txt"
uttblist_idx_test="uttblist_idx"

asr_config=conf/contextual/whisper/train_asr_whisper_medium_contextual_adapter_decoder.yaml
inference_config=conf/contextual/whisper/decode_asr_whisper_prefix_tuning_c20.yaml
asr_tag=whisper/run_medium_multilateinteraction_contextual_adapter_decoder

pretrained_model=../asr1/exp/asr_whisper_medium_lora_decoder/3epoch.pth

lm_config=conf/exp/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

if [ ! -f "data/train/token.man.2" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/token.man.2 does not exist! Run from stage=1 again."
        exit 1
    fi
fi

man_chars=3955
bpe_nlsyms=""

source data/train/token.man.2  # for bpe_nlsyms & man_chars
# nbpe=$((3000 + man_chars + 4))  # 5626
nbpe=5000
# English BPE: 3000 / Mandarin: 2622 / other symbols: 4

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 20 \
    --gpu_inference false \
    --inference_nj 10 \
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
    --lm_fold_length 150 \
    --pretrained_model "${pretrained_model},${pretrained_model}:decoder.decoders.token_embedding:contextualizer.encoder.embed" \
    --ignore_init_mismatch true \
    "$@"

