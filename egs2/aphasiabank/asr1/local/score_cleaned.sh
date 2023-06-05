#!/usr/bin/env bash

# ASR scoring, with [APH]/[NONAPH] tags removed, outputs overall CER/WER and CER/WER for APH and NONAPH subsets

set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
decode_dir=

log "$0 $*"
. utils/parse_options.sh

help_message=$(
  cat <<EOF
Usage: $0

Options:
    --decode_dir (string): Path to decoded results.
      For example, --decode_dir=exp/asr_train_asr_conformer_raw_en_char_sp/decode_asr_model_valid.acc.ave/test"
EOF
)

if [ -z "${decode_dir}" ]; then
  log "${help_message}"
  exit 2
fi

for sub in score_cer score_wer; do
  mkdir -p "${decode_dir}/${sub}_clean"
done

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for token_type in cer wer; do
    _dir="${decode_dir}/score_${token_type}_clean"

    # Filter speakers to get APH and NONAPH subsets
    cp -r "${decode_dir}/score_${token_type}"/{hyp.trn,ref.trn} ${_dir}
    python local/clean_score_dir.py "${decode_dir}/score_${token_type}" ${_dir}

    # Remove tags
    python local/clean_hyp_annotations.py "${_dir}/hyp.trn" "${_dir}/hyp.trn.clean"
    python local/clean_hyp_annotations.py "${_dir}/ref.trn" "${_dir}/ref.trn.clean"

    # for sub in en fr aph nonaph en.aph en.nonaph fr.aph fr.nonaph; do
    for sub in aph nonaph; do
      # Subset CER
      python local/clean_hyp_annotations.py "${_dir}/hyp.${sub}.trn" "${_dir}/hyp.${sub}.trn.clean"
      python local/clean_hyp_annotations.py "${_dir}/ref.${sub}.trn" "${_dir}/ref.${sub}.trn.clean"
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for token_type in cer wer; do
    _dir="${decode_dir}/score_${token_type}_clean"

    log "\n\nOverall ${token_type}"
    sclite -r "${_dir}/ref.trn.clean" -h "${_dir}/hyp.trn.clean" -i rm -o all stdout >"${_dir}/result.txt"
    grep -e Avg -e SPKR -m 2 "${_dir}/result.txt"

    # for sub in en fr aph nonaph en.aph en.nonaph fr.aph fr.nonaph; do
    for sub in aph nonaph; do
      log "\n\n${sub} ${token_type}"
      sclite -r "${_dir}/ref.${sub}.trn.clean" -h "${_dir}/hyp.${sub}.trn.clean" -i rm -o all stdout >"${_dir}/result.${sub}.txt"
      grep -e Avg -e SPKR -m 2 "${_dir}/result.${sub}.txt"
    done
  done
fi
