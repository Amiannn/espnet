import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

filter_pho = ['˥˩', '˧˥', '˥˥', '˨˩˦', '˧', '▁']
def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

def get_phoneme_count(datas):
    counts = {}
    for phonemes in datas:
        for phoneme in phonemes.split(' '):
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

def plot_freq_count(output_path, x, data, data1, indexis, ratios, tag=''):
    fig, ax1 = plt.subplots(figsize=(20, 8))             
    x_label = range(0, len(data), 1)

    index = list(range(len(data)))
    color = 'tab:blue'
    # ax1.plot(index, data, color=color, alpha=0.5, label="phoneme occurence (train-set)")
    ax1.fill_between(index, 0, data, color=color, alpha=0.5, linewidth=5, label="Train-set")
    # ax1.hist(data, color=color, alpha=0.5, bins=1000, label="phoneme occurence (train-set)")
    # ax.set_title(f'phoneme Occurrence - {tag}')
    ax1.set_xlabel('Phoneme (IPA)')
    ax1.set_ylabel('Occurence ($log_{2}$)', color=color)
    ax1.set_xticks(ticks=x_label, labels=x)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(color='gray', linestyle = '--', linewidth=1)
    ax1.set_ylim(ymin=0, ymax=24)
    ax1.set_xlim(xmin=0, xmax=len(data))

    color = 'tab:orange'
    ax1.fill_between(index, 0, data1, color=color, alpha=0.5, linewidth=5, label="Test-set")
    ax1.legend(loc=2)
    
    ax2   = ax1.twinx() 
    color = 'tab:gray'
    ax2.plot(indexis, ratios, linestyle = '--', color=color, linewidth=3, label="Segmented Cross-Entropy analysis")
    ax2.set_ylabel('Cross-Entropy', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0, ymax=10)
    ax2.legend(loc=1)

    plt.tight_layout()
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}_train_vs_test.png')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}_train_vs_test.svg')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}_train_vs_test.pdf')
    plt.savefig(out_path)

def freq_thresholding(datas, threshold):
    datas = {key: datas[key] for key in datas if datas[key] <= threshold}
    return datas

def count_cover_rate(all_phoneme_counts, phoneme_counts_subset):
    total_counts        = sum(list(all_phoneme_counts.values()))
    total_subset_counts = sum(list(phoneme_counts_subset.values()))
    return (total_subset_counts / total_counts), total_counts, total_subset_counts

train_text_path    = "./exp/analysis/pho_result_train.json"
test_text_path     = "./exp/analysis/pho_result.json"
dump_path          = "./exp/analysis"

train_text_datas = read_json(train_text_path)['pho']
test_text_datas  = [d[0] for d in read_json(test_text_path)['pho']]

train_counts = get_phoneme_count(train_text_datas)
test_counts  = get_phoneme_count(test_text_datas)

x = list(train_counts.keys())
print(x)
print(len(x))

# top 100
train_counts_log = {key:np.log2(train_counts[key]) for key in train_counts}
test_counts_log  = {key:(np.log2(test_counts[key]) if key in test_counts else 0) for key in train_counts}

train_length = sum([train_counts[key] for key in train_counts])
test_length  = sum([test_counts[key] for key in test_counts])

train_prob = {key:train_counts[key] / train_length for key in train_counts}
test_prob  = {key:(test_counts[key] if key in test_counts else 0.1) / test_length for key in train_counts}

step   = 5
index  = [i for i in range(0, (len(train_counts) // step + 1) * step, step)]
ratios = []

start = 0
train_values = list(train_prob.values())
test_values  = list(test_prob.values())
for i in index[1:]:
    train_value_sub = np.array([train_values[j] for j in range(start, i)])
    test_value_sub  = np.array([test_values[j] for j in range(start, i)])

    # re-normalize
    train_value_sub = train_value_sub / np.sum(train_value_sub)
    test_value_sub  = test_value_sub / np.sum(test_value_sub)

    ratio = -1 * np.sum((train_value_sub * np.log(test_value_sub)))
    ratios.append(ratio)
    start += step

indeixs = np.array(index) + step // 2
indeixs = np.concatenate([np.array([0]), indeixs])
ratios  = [0] + ratios

plot_freq_count(
    dump_path,
    x,
    list(train_counts_log.values()), 
    list(test_counts_log.values()),
    indeixs[:-1],
    ratios,
    tag='libri'
)

