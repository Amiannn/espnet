#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=1

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

other_text=data/local/other_text/text


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'CSJDATATOP' of db.sh"
    exit 1
fi

if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'CSJDATATOP' of db.sh"
    exit 1
fi

if [ ! -e "${dirha_folder}" ]; then
    log "Fill the value of 'CSJDATATOP' of db.sh"
    exit 1
fi

#if [ -z "${mic}" ]; then
#    log "Fill the value of 'mic' of db.sh"
#    exit 1
#fi
mic=Beam_Circular_Array
#mic="Beam_Circular_Array"  # Beam_Circular_Array Beam_Linear_Array KA6 L1C

dirha_wsj_folder=/export/b18/ruizhili/data/Data_processed # output folder for augmented wsj data and dirha data
#IR_folder=/export/b18/xwang/data/DIRHA_English_phrich_released_june2016_realonly_last/Data/Training_IRs # folders for Impulse responses for WSJ contamination
#sph_reader=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe
WSJ1=/export/corpora5/LDC/LDC94S13B

train_set=train_si284_$mic
train_test=dirha_real_$mic
train_dev=dirha_sim_$mic
recog_set=dirha_real_$mic

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"
    
    #expdir=exp/prepare_dirha_wsj_data_${mic}
    #$train_cmd $expdir/Data.log \
    #matlab -nodisplay -nosplash -r "addpath('./local/tools'); Data_Contamination('$mic','$WSJ0', '$WSJ1', '$dirha_folder', '$dirha_wsj_folder', '$IR_folder', '$sph_reader');exit"
    
    # augmented train
    wsj0_contaminated_folder=WSJ0_contaminated_mic_$mic # path of the wsj0 training data
    wsj1_contaminated_folder=WSJ1_contaminated_mic_$mic # path of the wsj1 training data
    local/wsj_data_prep.sh ${dirha_wsj_folder}/$wsj0_contaminated_folder/??-{?,??}.? ${dirha_wsj_folder}/$wsj1_contaminated_folder/??-{?,??}.? || exit 1;
    local/wsj_format_data.sh $mic || exit 1;

    # driha test
    DIRHA_wsj_data=${dirha_wsj_folder}/DIRHA_wsj_oracle_VAD_mic_$mic # path of the test data
    local/dirha_data_prep.sh $DIRHA_wsj_data/Sim dirha_sim_$mic  || exit 1;
    local/dirha_data_prep.sh $DIRHA_wsj_data/Real dirha_real_$mic  || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Srctexts preparation"

    mkdir -p "$(dirname ${other_text})"

    # NOTE(kamo): Give utterance id to each texts.
    zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
	    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
	    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
