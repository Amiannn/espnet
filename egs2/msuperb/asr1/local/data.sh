#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
nlsyms_txt=data/local/nlsyms.txt
duration=10min # duration can be either 10min or 1h
multilingual=true
lid=false
single_lang=eng # lang for single lang data preparation 
                # candidates: eng, deu, rus, pol, swe, jpn, cmn, sat, nob, xty

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MSUPERB}
if [ -z "${MSUPERB}" ]; then
    log "Fill the value of 'MSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
    log "stage1: Download data to ${MSUPERB}"
    log "Not released yet"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for multilingual SUPERB"

    if "${multilingual}"; then
        mkdir -p data/train_${duration}
        mkdir -p data/dev_${duration}
        mkdir -p data/test_${duration}
 
        python local/data_prep.py \
            --train_set train_${duration} \
            --train_dev dev_${duration} \
            --test_set test_${duration} \
            --duration ${duration} \
            --source ${MSUPERB} \
            --lid ${lid}

        for x in "train" "dev" "test"; do
            utils/utt2spk_to_spk2utt.pl \
                data/${x}_${duration}/utt2spk \
                > data/${x}_${duration}/spk2utt
            utils/fix_data_dir.sh data/${x}_${duration}
        done
    else
        for x in "train" "dev" "test"; do
            mkdir -p data/${x}_${duration}_${single_lang}
        done
        
        python local/single_lang_data_prep.py \
            --duration ${duration} \
            --source ${MSUPERB} \
            --lid ${lid} \
            --lang ${single_lang}

        for x in "train" "dev" "test"; do
             utils/utt2spk_to_spk2utt.pl \
                 data/${x}_${duration}_${single_lang}/utt2spk \
                 > data/${x}_${duration}_${single_lang}/spk2utt
             utils/fix_data_dir.sh data/${x}_${duration}_${single_lang}
        done
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage3: Create non-linguistic symbols for language ID"
    if "${multilingual}"; then
        train_set=data/train_${duration}
    else
        train_set=data/train_${duration}_${single_lang}
    fi
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms_txt}
    log "save non-linguistic symbols in ${nlsyms_txt}"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
