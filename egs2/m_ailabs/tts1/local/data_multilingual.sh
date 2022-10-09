#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

# silence part trimming related
do_trimming=false

# token type ("char" or "phn")
token_type=phn

# dataset configuration
db_root=/home/saeki/workspace/ssd1/mailab

# locale all
langs=("de_DE" "en_UK" "it_IT" "es_ES" "en_US" "fr_FR" "uk_UK" "ru_RU" "pl_PL")

whole_set_all=""
train_set_all=""
dev_set_all=""
eval_set_all=""

suffix=""
if [ "${token_type}" = phn ]; then
    suffix="_phn"
fi

for lang in ${langs[@]}; do

    # Listing all the speakers for each locale
    if [ ${lang} = "de_DE" ]; then
        spkrs=("angela" "rebecca" "ramona" "eva" "karlsson")
        g2p="espeak_ng_german"
    elif [ ${lang} = "en_UK" ]; then
        spkrs=("elizabeth")
        g2p="g2p_en"
    elif [ ${lang} = "it_IT" ]; then
        spkrs=("lisa" "riccardo")
        g2p="espeak_ng_italian"
    elif [ ${lang} = "es_ES" ]; then
        spkrs=("karen" "tux" "victor")
        g2p="espeak_ng_spanish"
    elif [ ${lang} = "en_US" ]; then
        spkrs=("judy" "mary" "elliot")
        g2p="g2p_en"
    elif [ ${lang} = "fr_FR" ]; then
        spkrs=("ezwa" "nadine" "bernard" "gilles" "zeckou")
        g2p="espeak_ng_french"
    elif [ ${lang} = "uk_UK" ]; then
        spkrs=("sumska" "loboda" "miskun" "obruchov" "shepel")
        # Using russian g2p because phonemizer does not support uk
        g2p="espeak_ng_russian"
    elif [ ${lang} = "ru_RU" ]; then
        spkrs=("hajdurova" "minaev" "nikolaev")
        g2p="espeak_ng_russian"
    elif [ ${lang} = "pl_PL" ]; then
        spkrs=("nina" "piotr")
        g2p="espeak_ng_polish"
    else
        echo "${lang} is not supported."
        exit 1
    fi

    # Preparing the dataset for each locale and speaker
    for spkr in ${spkrs[@]}; do

        # org set, and train, dev, eval sets with suffix
        org_set=${lang}_${spkr}
        whole_set="${org_set}${suffix}"
        train_set="${whole_set}_train"
        dev_set="${whole_set}_dev"
        eval_set="${whole_set}_eval"
        # Appending each data dir to combine them later
        whole_set_all="${whole_set_all} data/${whole_set}"
        train_set_all="${train_set_all} data/${train_set}"
        dev_set_all="${dev_set_all} data/${dev_set}"
        eval_set_all="${eval_set_all} data/${eval_set}"

        if [ -d data/${org_set} ]; then
            echo "Skipping ${org_set}"
            continue
        fi

        echo "Processing ${lang}_${spkr}."
        local/data_prep.sh ${db_root} ${lang} ${spkr} data/${org_set}
        utils/fix_data_dir.sh data/${org_set}
        utils/validate_data_dir.sh --no-feats data/${org_set}

        # Trim silence parts at the beginning and the end of audio
        if ${do_trimming}; then
            echo "Trimmng silence."
            scripts/audio/trim_silence.sh \
                --cmd "${train_cmd}" \
                --nj 32 \
                --fs 16000 \
                --win_length 1024 \
                --shift_length 256 \
                --threshold 60 \
                --min_silence 0.01 \
                data/${org_set} \
                data/${org_set}/log
        fi

        if [ "${token_type}" = phn ]; then
            log "pyscripts/utils/convert_text_to_phn.py"
            utils/copy_data_dir.sh "data/${org_set}" "data/${org_set}_phn"
            pyscripts/utils/convert_text_to_phn.py \
                --g2p "${g2p}" --nj 1 \
                "data/${org_set}/text" "data/${org_set}_phn/text"
            utils/fix_data_dir.sh "data/${org_set}_phn"
        fi

        echo "Deviding into subsets."
        # make a dev set
        utils/subset_data_dir.sh --last data/${whole_set} 50 data/${whole_set}_tmp
        utils/subset_data_dir.sh --last data/${whole_set}_tmp 25 data/${dev_set}
        utils/subset_data_dir.sh --first data/${whole_set}_tmp 25 data/${eval_set}
        n=$(($(wc -l <data/${whole_set}/wav.scp) - 50))
        utils/subset_data_dir.sh --first data/${whole_set} ${n} data/${train_set}
        rm -rf data/${whole_set}_tmp

    done
done

# Combining all the sets
utils/data/combine_data.sh data/whole${suffix} ${whole_set_all}
utils/data/combine_data.sh data/tr_no_dev${suffix} ${train_set_all}
utils/data/combine_data.sh data/dev${suffix} ${dev_set_all}
utils/data/combine_data.sh data/eval${suffix} ${eval_set_all}

# Delete original filders
rm -rf ${whole_set_all}
rm -rf ${train_set_all}
rm -rf ${dev_set_all}
rm -rf ${eval_set_all}
log "Successfully finished. [elapsed=${SECONDS}s]"
