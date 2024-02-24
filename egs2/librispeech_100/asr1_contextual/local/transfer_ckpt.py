import os
import torch

MODEL_B_PATH = "./exp/asr_train_rnnt_std_tcpgen_finetune_raw_en_bpe600_use_wandbtrue_sp_suffix/valid.loss.best.pth"
MODEL_A_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/valid.loss.ave_10best.pth"

OUTPUT_B_PATH  = "/share/nas165/amian/experiments/speech/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_rnnt_std_tcpgen_finetune_raw_en_bpe600_use_wandbtrue_sp_suffix"
OUTPUT_A_PATH  = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix"

model_a = torch.load(MODEL_A_PATH)
model_b = torch.load(MODEL_B_PATH)

print(model_b.keys())
print('-' * 30)
print(model_a.keys())

b_keys = list(model_b.keys())
a_keys = list(model_a.keys())

same_keys = []
for key in a_keys:
    if key in b_keys:
        print(f'same: {key}')
        same_keys.append(key)
    else:
        print(f'diff a: {key}')

for key in b_keys:
    if key not in same_keys:
        print(f'diff b: {key}')

"""
"joint_network.lin_biasing.weight"
"joint_network.lin_biasing.bias"
"Qproj_acoustic.weight"
"Qproj_acoustic.bias"
"Qproj_char.weight"
"Qproj_char.bias"
"Kproj.weight"
"Kproj.bias"
"ooKBemb.weight"
"pointer_gate.weight"
"pointer_gate.bias"

"rareword.ooKBemb.weight"        : "ooKBemb.weight",
"rareword.Qproj_acoustic.weight" : "Qproj_acoustic.weight",
"rareword.Qproj_acoustic.bias"   : "Qproj_acoustic.bias",
"rareword.Qproj_semantic.weight" : "Qproj_char.weight",
"rareword.Qproj_semantic.bias"   : "Qproj_char.bias",
"rareword.Kproj.weight"          : "Kproj.weight",
"rareword.Kproj.bias"            : "Kproj.bias",
"rareword.Vproj.weight"          : "Kproj.weight",
"rareword.Vproj.bias"            : "Kproj.bias",
"rareword.gate.weight"           : "pointer_gate.weight",
"rareword.gate.bias"             : "rareword.gate.bias",
"rareword.dbias.weight": "joint_network.lin_biasing.weight",
"rareword.dbias.bias"  : "joint_network.lin_biasing.bias",
"""

b2a = {
    "rareword.ooKBemb.weight"       : "ooKBemb.weight",
    "rareword.Qproj_acoustic.weight": "Qproj_acoustic.weight",
    "rareword.Qproj_acoustic.bias"  : "Qproj_acoustic.bias",
    "rareword.Qproj_semantic.weight": "Qproj_char.weight",
    "rareword.Qproj_semantic.bias"  : "Qproj_char.bias",
    "rareword.Kproj.weight"         : "Kproj.weight",
    "rareword.Kproj.bias"           : "Kproj.bias",
    "rareword.Vproj.weight"         : "Kproj.weight",
    "rareword.Vproj.bias"           : "Kproj.bias",
    "rareword.gate.weight"          : "pointer_gate.weight",
    "rareword.gate.bias"            : "pointer_gate.bias",
    "rareword.dbias.weight"         : "joint_network.lin_biasing.weight",
    "rareword.dbias.bias"           : "joint_network.lin_biasing.bias",
}

a2b = {b2a[key]:key for key in b2a}

transfered_b_ckpt = {}

for key in b_keys:
    if key in a_keys:
        data = model_a[key]
    else:
        data = model_a[b2a[key]]
    transfered_b_ckpt[key] = data

output_path = os.path.join(OUTPUT_B_PATH, 'transfered.pth')
torch.save(transfered_b_ckpt, output_path)

transfered_a_ckpt = {}

for key in a_keys:
    if key in b_keys:
        data = model_b[key]
    else:
        data = model_b[a2b[key]]
    transfered_a_ckpt[key] = data

output_path = os.path.join(OUTPUT_A_PATH, 'transfered.pth')
torch.save(transfered_a_ckpt, output_path)