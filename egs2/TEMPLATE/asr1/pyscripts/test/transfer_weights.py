import os
import torch

def print_model(ckpt):
    for name in ckpt:
        print(name)
    print()

def modify_names(ckpt):
    new_ckpt = {}
    for name in ckpt:
        new_name = name.replace('encoders.', '')
        new_name = new_name.replace('decoders.', '')
        new_ckpt[new_name] = ckpt[name]
    return new_ckpt

def transfer_weights(A, B):
    for name in A:
        A[name] = B[name]

def filter_out(ckpt, names):
    new_ckpt = {}
    for name in ckpt:
        if any(name.find(n) != -1 for n in names):
            new_ckpt[name] = ckpt[name]
    return new_ckpt

# whisper_model_path = "ckpts/medium.pt"
model_path = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix/valid.loss.ave_10best.pth"
model_ckpt = torch.load(model_path, map_location=torch.device("cpu"))
# print_model(model_ckpt)

contextualizer_ckpt = filter_out(model_ckpt, ['ctc.', 'contextualizer.'])
print_model(contextualizer_ckpt)

output_dir = "/".join(model_path.split('/')[:-1])
output_path = os.path.join(output_dir, 'xdotretriever.pt')
torch.save(contextualizer_ckpt, output_path)

# whisper_model_ckpt = torch.load(whisper_model_path, map_location=torch.device("cpu"))
# transfer_weights(whisper_model_ckpt['model_state_dict'], stage1_model_ckpt)
# output_path = os.path.join('transfered_ckpts', 'openai_stage1.pt')
# torch.save(whisper_model_ckpt, output_path)