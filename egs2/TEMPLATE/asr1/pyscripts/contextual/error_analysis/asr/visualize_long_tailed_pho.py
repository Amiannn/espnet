import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

filter_pho = ['˥˩', '˧˥', '˥˥', '˨˩˦', '˧', '▁']

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

def get_word_count(datas):
    counts = {}
    for text in datas:
        phonemes = text.split(' ')
        for phoneme in phonemes:
            if phoneme in filter_pho:
                    continue
            phoneme = phoneme.replace("ˈ", '')
            phoneme = phoneme.replace("ˌ", '')
            phoneme = phoneme.replace("ː", '')
            phoneme = phoneme.replace("ʰ", '')
            counts[phoneme] = counts[phoneme] + 1 if phoneme in counts else 1
    counts = sorted([[counts[key], key] for key in counts], reverse=True)
    counts = {data[1]: data[0] for i, data in enumerate(counts)}
    return counts

def plot_word_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 5))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Word Counts - {tag}')
    plt.xlabel('Word')
    plt.ylabel('Counts')
    # plt.xticks(range(max(data.keys()) + 1), rotation=90)
    plt.xticks(rotation=90)

    out_path = os.path.join(output_path, f'word_counts_{tag}.png')
    plt.savefig(out_path)

def plot_freq_count(output_path, x, data, tag=''):
    fig, ax1 = plt.subplots(figsize=(20, 8))             
    x_label = range(0, len(data), 1)

    index = list(range(len(data)))
    color = 'tab:blue'
    line1 = ax1.plot(index, data, color=color, linewidth=5, label="Phoneme occurence")
    # ax.set_title(f'Word Occurrence - {tag}')
    ax1.set_xlabel('Phoneme (IPA)')
    ax1.set_ylabel('Occurence ($log_{2}$)', color=color)
    ax1.set_xticks(ticks=x_label, labels=x)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(color='gray', linestyle = '--', linewidth=1)
    ax1.legend(loc=2)
    ax1.set_ylim(ymin=0, ymax=24)
    ax1.set_xlim(xmin=0, xmax=len(data)-1)

    # ax1.plot(entity_x, entity_y, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_path, f'phone_plot_counts_{tag}.png')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phone_plot_counts_{tag}.svg')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phone_plot_counts_{tag}.pdf')
    plt.savefig(out_path)

def freq_thresholding(datas, threshold):
    datas = {key: datas[key] for key in datas if datas[key] <= threshold}
    return datas

def count_cover_rate(all_word_counts, word_counts_subset):
    total_counts        = sum(list(all_word_counts.values()))
    total_subset_counts = sum(list(word_counts_subset.values()))
    return (total_subset_counts / total_counts), total_counts, total_subset_counts

train_text_path    = "./exp/analysis/pho_result_train.json"
dump_path          = "./exp/analysis"

train_text_datas = read_json(train_text_path)['pho']
counts           = get_word_count(train_text_datas)

counts_log  = {key:np.log2(counts[key]) for key in counts}
x = list(counts.keys())
print(x)
print(len(x))
plot_freq_count(dump_path, x, list(counts_log.values()), f'phone_occurance')
