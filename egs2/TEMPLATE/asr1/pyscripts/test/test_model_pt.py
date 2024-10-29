import os
import torch


model_lora_ckpt_path = "/mnt/storage1/experiments/espnet/egs2/esun/asr1/exp/asr_whisper_medium_lora_decoder/3epoch.pth"
model_ckpt_path = "/mnt/storage1/experiments/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_contextual_adapter_decoder/140epoch.pth"

state_lora_dict = torch.load(model_lora_ckpt_path, map_location=torch.device("cpu"))
state_dict = torch.load(model_ckpt_path, map_location=torch.device("cpu"))

print(f'state_lora_dict: {state_lora_dict.keys()}')
print(f'_' * 30)
print(f'state_dict: {state_dict.keys()}')

new_state_dict = {}

for key in state_lora_dict:
    new_state_dict[key] = state_lora_dict[key]

for key in state_dict:
    if 'contextual' in key:
        print(key)
        new_state_dict[key] = state_dict[key]

output_path = model_ckpt_path.replace('.pth', '_fixed.pth')
torch.save(new_state_dict, output_path)