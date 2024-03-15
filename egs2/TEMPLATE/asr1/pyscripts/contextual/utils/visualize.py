import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_attention_map(
    frame2align,
    attention,
    text,
    labels,
    debug_path,
    uttid='test',
):
    attention = torch.flip(attention, [0, 2])
    attention = attention.squeeze(0).T.detach().cpu().resolve_conj().resolve_neg().numpy()
    xlabels   = [
        f'{frame2align[i]} {i}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
    ]

    labels = [f'{labels[len(labels) - i - 1]}' for i in range(len(labels))]
    plt.rcParams.update({'font.size': 8})

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(45, 10))
    axes.xaxis.set_ticks(np.arange(0, attention.shape[1], 1))
    axes.yaxis.set_ticks(np.arange(0, attention.shape[0], 1))
    axes.set_xticks(np.arange(-.5, attention.shape[1], 10), minor=True)
    axes.set_yticks(np.arange(-.5, attention.shape[0], 1), minor=True)
    axes.set_xticklabels(xlabels, rotation=90)
    axes.set_yticklabels(labels)

    axes.imshow(attention, aspect='auto')
    axes.grid(which='minor', color='w', linewidth=0.5, alpha=0.3)
    plt.title(text)
    output_path = os.path.join(debug_path, f'{uttid}_attention_map.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()