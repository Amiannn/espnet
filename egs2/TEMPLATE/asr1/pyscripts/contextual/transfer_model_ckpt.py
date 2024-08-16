import os
import torch

model_path = "./exp/asr_whisper/run_medium_xphoneRetriever_prompting/0epoch.pth"

model_state_dict = torch.load(model_path)

transfer_state_dict = {}
for key in model_state_dict:
    if 'decoder.decoders' in key:
        _key = key.replace('decoder.decoders', 'decoder')
    elif 'encoder.encoders' in key:
        _key = key.replace('encoder.encoders', 'encoder')
    transfer_state_dict[_key] = model_state_dict[key]

print('_' * 30)

whisper_model_path = "./medium.pt"
whisper_model_state_dict = torch.load(whisper_model_path)
new_whisper_model_state_dict = {
    "dims": whisper_model_state_dict["dims"],
    "model_state_dict": {},
}
for key in whisper_model_state_dict["model_state_dict"]:
    new_whisper_model_state_dict["model_state_dict"][key] = transfer_state_dict[key]

out_path = "/".join(model_path.split("/")[:-1])
debug_out_path = os.path.join(out_path, 'transfered_model_weights')
if not os.path.isdir(debug_out_path):
    os.makedirs(debug_out_path)

filename = model_path.split('/')[-1]
output_path = os.path.join(debug_out_path, f'transfer_{filename}')
torch.save(new_whisper_model_state_dict, output_path)

print(f'_' * 30)
test = torch.load(output_path)
print(test.keys())
print(test['model_state_dict'].keys())

"""
0. python3 -m pyscripts.contextual.transfer_model_ckpt 
1. python3 convert_openai_to_hf.py --checkpoint_path /home/ubuntu/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_xphoneRetriever_prompting/transfered_model_weights/transfer_0epoch.pth  --pytorch_dump_folder_path /home/ubuntu/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_xphoneRetriever_prompting/transfered_model_weights/tf
2. ct2-transformers-converter --model . --output_dir /home/ubuntu/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_xphoneRetriever_prompting/transfered_model_weights/ct --copy_files tokenizer.json preprocessor_config.json --quantization float16
"""