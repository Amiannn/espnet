#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=44k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=

python local/prep_data.py --clarity_root /raid/users/popcornell/Clarity/target_dir/clarity_CEC1_data/clarity_data/

train_set=train
valid_set=dev
test_sets="dev"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --spk_num 1 \
    --ref_channel 0 \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 1 --stop-stage 2" \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
