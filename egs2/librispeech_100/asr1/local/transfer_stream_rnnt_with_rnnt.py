import os
import torch

# MODEL_RNNT_CKPT           = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_mini_raw_en_bpe600_sp_suffix/43epoch.pth"
MODEL_RNNT_CKPT           = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1/exp/asr_train_rnnt_conformer_ngpu4_raw_en_bpe5000_sp/valid.loss.ave_5best.pth"
MODEL_RNNT_STREAMING_CKPT = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_transducer_conformer_finetune_raw_en_bpe5000_sp/1epoch.pth"

rnnt_datas           = torch.load(MODEL_RNNT_CKPT)
rnnt_streaming_datas = torch.load(MODEL_RNNT_STREAMING_CKPT)

ckpt_name  = MODEL_RNNT_CKPT.split('/')[-1]
output_dir = "/".join(MODEL_RNNT_CKPT.split('/')[:-1])
new_ckpt_name = f'transfered.{ckpt_name}'

# print(f'RNN-T Datas:')
# for key in rnnt_datas:
#     print(key)
# print(f'_' * 50)

# print(f'RNN-T Streaming Datas:')
# for key in rnnt_streaming_datas:
#     print(key)
# print(f'_' * 50)

new_ckpt = {}
for key_rnnt, key_stream in zip(rnnt_datas, rnnt_streaming_datas):
    new_ckpt[key_stream] = rnnt_datas[key_rnnt]

output_path = os.path.join(output_dir, new_ckpt_name)
torch.save(new_ckpt, output_path)