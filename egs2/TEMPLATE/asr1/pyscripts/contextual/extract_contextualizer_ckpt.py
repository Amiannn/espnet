import os
import torch

model_path = "/home/ubuntu/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_xphone_retrieval_balanced/valid.loss.ave_10best.pth"

model_state_dict = torch.load(model_path)

state_dict = {}
for key in model_state_dict:
    if 'contextualizer' in key:
        _key = key.replace('contextualizer.', '')
        state_dict[_key] = model_state_dict[key]

out_path = "/".join(model_path.split("/")[:-1])
debug_out_path = os.path.join(out_path, 'transfered_model_weights')
if not os.path.isdir(debug_out_path):
    os.makedirs(debug_out_path)

filename = model_path.split('/')[-1]
output_path = os.path.join(debug_out_path, f'contextualizer_{filename}')
torch.save(state_dict, output_path)
