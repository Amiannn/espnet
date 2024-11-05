#!/bin/bash

# Set default paths (modify these paths according to your environment)
SPM_PATH="whisper_multilingual"
CONTEXT_SPM_PATH=$SPM_PATH
TOKEN_PATH="./data/zh_token_list/whisper_multilingual/tokens.txt"
CONTEXT_TOKEN_PATH=$TOKEN_PATH
MODEL_CONF="./conf/contextual/whisper/train_asr_whisper_medium_contextual_adapter_decoder_pretrained.yaml"
MODEL_PATH="./exp/asr_whisper/run_medium_contextual_adapter_decoder_pretrained_ce/122epoch.pth"
MODEL_LORA_PATH="../asr1/exp/asr_whisper_medium_lora_decoder/3epoch.pth"
STATS_PATH=None
RAREWORD_PATH="./local/contextual/rarewords/esun.entity.txt"
SPEECH_SCP_PATH="./dump/raw/test/wav.scp"
CONTEXT_LIST_PATH="./dump/raw/test/uttblist_idx_entity"
CONTEXT_LIST_XPHONE_PATH="./local/contextual/ssl_features/esun.entity.xphone.seq.pt"
REFERENCE_PATH="./data/test/text"
TOKEN_TYPE="whisper_multilingual"
CONTEXT_TOKEN_TYPE=$TOKEN_TYPE
USE_LOCAL_ATTN_CONV="false"  # Set to "--use_local_attn_conv" if needed
MAX_COUNT=20

# Run the inference script
python3 -m pyscripts.contextual.inference_whisper \
    --spm_path $SPM_PATH \
    --context_spm_path $CONTEXT_SPM_PATH \
    --token_path $TOKEN_PATH \
    --context_token_path $CONTEXT_TOKEN_PATH \
    --model_conf $MODEL_CONF \
    --model_path $MODEL_PATH \
    --model_lora_path $MODEL_LORA_PATH \
    --stats_path $STATS_PATH \
    --rareword_path $RAREWORD_PATH \
    --speech_scp_path $SPEECH_SCP_PATH \
    --context_list_path $CONTEXT_LIST_PATH \
    --context_list_xphone_path $CONTEXT_LIST_XPHONE_PATH \
    --reference_path $REFERENCE_PATH \
    --token_type $TOKEN_TYPE \
    --context_token_type $CONTEXT_TOKEN_TYPE \
    --max_count $MAX_COUNT
