#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh
. ./path.sh
. ./cmd.sh

cmd=run.pl
ngpu=4 # number of GPUs to extract features
stage=1
stop_stage=4
model_conf=$1

echo "We use AV-HuBERT ${model_conf} configuration"

log "$0 $*"
. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -z "${LRS3}" ]; then
        log "Please download dataset 'https://mmai.io/datasets/lip_reading/' & Fill the value of 'LRS3' of db.sh"
        exit 1
    fi

    if [ ! -e data/test/text ]; then
        python ./local/scp_gen.py --data_dir ${LRS3} --model ${model_conf}
    fi

    for dataset in train val test; do
        if [ -e data/${dataset}/spk2utt ]; then
            continue
        fi
        cp data/${dataset}/video.scp  data/${dataset}/wav.scp
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/utt2spk
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/spk2utt
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Download the pretrained AV-HuBERT model to extract visual features from official AV-HuBERT repository.
    # https://facebookresearch.github.io/av_hubert/
    mkdir -p local/pre-trained
    if [ ! -f local/pre-trained/${model_conf}_vox_iter5.pt ]; then
        echo "Download pre-trained model noise-pretrain/${model_conf}_vox_iter5.pt from https://facebookresearch.github.io/av_hubert/"
        echo "If the download continues to fail, download it manually from the site."

        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/${model_conf}_vox_iter5.pt -O local/pre-trained/${model_conf}_vox_iter5.pt
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Download extracted facial landmark to preprocess from the following repository.
    # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
    if [ ! -e local/LRS3_landmarks ]; then
        if python -c "import gdown" &> /dev/null; then
            echo 'requirements installed'
        else
            echo 'please install required packages by run ". ./path.sh; pip install gdown"'
            exit 1;
        fi
        echo "Download extracted landmark from https://drive.google.com/uc?id=1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW"
        echo "If the download continues to fail, download it manually from the site & unzip at data/LRS3_landmarks."

        gdown https://drive.google.com/uc?id=1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW -O local/LRS3_landmarks.zip --continue
        unzip -qq local/LRS3_landmarks.zip
        rm local/LRS3_landmarks.zip
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Extract audio visual features and fuse the two features using pre-trained AV-HuBERT front-ends
    if python -c "import skvideo, skimage, cv2, python_speech_features" &> /dev/null; then
        echo 'requirements installed'
    else
        echo 'please install required packages by run ". ./path.sh; pip install sk-video scikit-image opencv-python python_speech_features"'
        exit 1;
    fi

    for dataset in train val test; do
        if [ -e data/${dataset}/feats.scp ]; then
            continue
        fi

        echo "extracting visual feature for [${dataset}]"
        log_dir=data/${dataset}/split_${ngpu}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $ngpu); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dataset}/video.scp $split_scps || exit 1

        $cmd JOB=1:$ngpu ${log_dir}/extract_av_feature.JOB.log python ./local/extract_av_feature.py \
            --file_list ${log_dir}/video.JOB.scp \
            --model ${model_conf} \
            --job JOB || exit 1

        for n in $(seq $ngpu); do
            cat ${log_dir}/feature.${n}.scp
        done > data/${dataset}/feats.scp

        for n in $(seq $ngpu); do
            cat ${log_dir}/num_frames.${n}.txt
        done > data/${dataset}/num_frames.txt

    done
    rm -rf data/temp

    for dataset in train val test; do
        utils/fix_data_dir.sh data/${dataset}
    done

fi
