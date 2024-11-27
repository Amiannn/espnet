import os
import torch

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

model_ckpt_path = "./exp/asr_whisper/run_medium_contextual_adapter_decoder_alpha0.9/45epoch.pth"

state_dict     = torch.load(model_ckpt_path, map_location=torch.device("cpu"))
new_state_dict = {}

for key in state_dict:
    if 'contextual' in key or 'lora' in key:
        print(key)
        new_state_dict[key] = state_dict[key]

output_path = model_ckpt_path.replace('.pth', '_fixed.pth')
torch.save(new_state_dict, output_path)