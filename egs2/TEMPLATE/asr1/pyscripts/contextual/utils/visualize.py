import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

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
        # f'{frame2align[i]} {i}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
        f'{frame2align[i]}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
    ]

    labels = [f'{labels[len(labels) - i - 1]}' for i in range(len(labels))]
    plt.rcParams.update({'font.size': 24})
    # plt.rcParams.update({'font.size': 12})

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(20, 5))
    # fig, axes = plt.subplots(1, 1, figsize=(45, 30))
    axes.xaxis.set_ticks(np.arange(0, attention.shape[1], 1))
    axes.yaxis.set_ticks(np.arange(0, attention.shape[0], 1))
    axes.set_xticks(np.arange(-.5, attention.shape[1], 10), minor=True)
    axes.set_yticks(np.arange(-.5, attention.shape[0], 1), minor=True)
    # axes.set_xticklabels(xlabels, rotation=90)
    axes.set_xticklabels(xlabels)
    axes.set_yticklabels(labels)

    axes.imshow(attention, aspect='auto')
    axes.grid(axis='y', which='minor', color='w', linewidth=0.5, alpha=0.3)
    # plt.title(text)
    output_path = os.path.join(debug_path, f'{uttid}_attention_map.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    output_path = os.path.join(debug_path, f'{uttid}_attention_map.svg')
    plt.savefig(output_path, bbox_inches="tight")
    plt.clf()

def plot_tsne(
        X,
        label,
        debug_path,
        uttid='test',
    ):
    # tsne = TSNE(n_components=2, verbose=1, perplexity=5)
    tsne = TSNE(n_components=2, verbose=1)
    X    = tsne.fit_transform(X.detach())

    fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    plt.scatter(x=X[:, 0], y=X[:, 1], s=10)

    if label is not None:
        texts = []
        # plt.rcParams.update({'font.size': 8})
        plt.rcParams.update({'font.size': 12})
        for i in range(X.shape[0]):
            x, y = X[i, 0], X[i, 1]
            texts.append(plt.text(x, y, label[i]))

    output_path = os.path.join(debug_path, f'{uttid}_tsne.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

def plot_gate(
    gate_prob,
    debug_path,
    uttid='test',
):
    plt.plot(gate_prob)
    output_path = os.path.join(debug_path, f'{uttid}_gate.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()