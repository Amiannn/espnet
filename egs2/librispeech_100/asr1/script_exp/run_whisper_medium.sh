train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/tuning/train_asr_whisper_full.yaml
inference_config=conf/decode_asr_whisper_noctc_greedy.yaml

asr_tag=whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --skip_data_prep true \
    --skip_train true \
    --skip_eval false \
    --gpu_inference false \
    --lang en \
    --ngpu 1 \
    --nj 4 \
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
    --inference_asr_model valid.acc.ave.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"