import os
import matplotlib.pyplot as plt

def plot_attention_map(
    model,
    speech,
    text,
    biasingwords,
    target, 
    logp, 
    attention,
    enc_embeds,
    cb_embeds,
    labels,
    cb_class,
    cb_types,
    uttid='test',
):
    alignments = alignment(
        logp, 
        target, 
        model.blank_id, 
        model.token_list, 
        speech[0]
    )
    attention   = attention.squeeze(0).T.detach().cpu().resolve_conj().resolve_neg().numpy()
    frame2align = {start: token for token, start, end in alignments}
    xlabels = [
        f'{frame2align[i]} {i}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
    ]
    print(f'xlabels: {len(xlabels)}')

    # plot_tsne(model, enc_embeds, cb_embeds, cb_class, cb_types, labels)
    # labels = [f'{labels[i]} {int(cb_class[i])}' for i in range(len(labels))]
    labels = [f'{labels[i]}' for i in range(len(labels))]
    plt.rcParams.update({'font.size': 8})

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(8, 2))
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