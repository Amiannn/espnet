#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/whisper/train_asr_whisper_lora_decoder_and_embed.yaml
inference_config=conf/whisper/decode_asr_whisper_noctc_greedy.yaml
asr_tag=whisper_medium_lora_decoder_bpe5000

lm_config=conf/exp/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

man_chars=3955
bpe_nlsyms=""

source data/train/token.man.2  # for bpe_nlsyms & man_chars
# nbpe=$((3000 + man_chars + 4))  # 5626
nbpe=5000
# English BPE: 3000 / Mandarin: 2622 / other symbols: 4


CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 20 \
    --gpu_inference false \
    --inference_nj 20 \
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
    --inference_asr_model 3epoch.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text.eng.bpe" \
    --score_opts "-e utf-8 -c NOASCII" \
    "$@"
