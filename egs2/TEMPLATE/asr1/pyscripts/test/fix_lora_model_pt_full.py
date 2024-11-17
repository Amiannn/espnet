import os
import torch

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def filter_weights(state_dict, prefixs):
    new_state_dict = {}
    for key in state_dict:
        if any([key.find(prefix) != -1 for prefix in prefixs]):
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def change_prefix(state_dict, target, prefix):
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace(target, prefix)
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

original_model_ckpt_path  = "./exp/asr_whisper/run_medium_rradapter/0epoch.pth"
adapter_model_ckpt_path   = "./exp/asr_whisper/run_medium_contextual_adapter_decoder_pretrained_ce/valid.loss.ave_10best_fixed.pth"
retriever_model_ckpt_path = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix/valid.loss.ave_10best.pth"

original_state_dict  = torch.load(original_model_ckpt_path, map_location=torch.device("cpu"))
adapter_state_dict   = torch.load(adapter_model_ckpt_path, map_location=torch.device("cpu"))
retriever_state_dict = torch.load(retriever_model_ckpt_path, map_location=torch.device("cpu"))

for key in original_state_dict:
    print(key)
    
# output_path = model_ckpt_path.replace('.pth', '_fixed.pth')
# torch.save(new_state_dict, output_path)

lora_state_dict      = filter_weights(adapter_state_dict, ['lora'])
adapter_state_dict   = filter_weights(adapter_state_dict, ['contextualizer'])
retriever_state_dict = filter_weights(retriever_state_dict, ['contextualizer'])

adapter_state_dict   = change_prefix(adapter_state_dict, 'contextualizer', 'contextualizer.adapter')
retriever_state_dict = change_prefix(retriever_state_dict, 'contextualizer', 'contextualizer.retriever')

new_state_dict = {}
# new_state_dict = original_state_dict
new_state_dict.update(lora_state_dict)
new_state_dict.update(adapter_state_dict)
new_state_dict.update(retriever_state_dict)

output_path = original_model_ckpt_path.replace('.pth', '_fixed.pth')
torch.save(new_state_dict, output_path)