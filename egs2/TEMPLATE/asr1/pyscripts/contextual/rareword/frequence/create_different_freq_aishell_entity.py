import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

def get_word_count(datas):
    counts = {}
    for uid, words in datas:
        for word in words:
            if len(word) < 2:
                continue
            counts[word] = counts[word] + 1 if word in counts else 1
    counts = sorted([[counts[key], key] for key in counts], reverse=True)
    counts = {data[1]: data[0] for i, data in enumerate(counts)}
    return counts

def plot_word_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 5))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Word Counts - {tag}')
    plt.xlabel('Word')
    plt.ylabel('Counts')
    plt.xticks(rotation=90)

    out_path = os.path.join(output_path, f'word_counts_{tag}.png')
    plt.savefig(out_path)

def plot_freq_count(output_path, data, data2, tag=''):
    fig, ax1 = plt.subplots(figsize=(20, 8))             
    x_label = range(0, len(data) + 1, 5000)

    index = list(range(len(data)))
    color = 'tab:blue'
    line1 = ax1.plot(index, data, color=color, linewidth=5, label="Word occurence")
    # ax.set_title(f'Word Occurrence - {tag}')
    ax1.set_xlabel('Sorted word index')
    ax1.set_ylabel('Occurence ($log_{2}$)', color=color)
    ax1.set_xticks(ticks=x_label, labels=x_label, rotation=45)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(color='gray', linestyle = '--', linewidth=1)
    ax1.legend(loc=2)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0, xmax=len(data))
    
    ax2   = ax1.twinx() 
    color = 'tab:orange'
    line2 = ax2.plot(index, data2, linestyle = '--', color=color, linewidth=5, alpha=0.5, label="Imbalance rate")
    ax2.set_ylabel('Context imbalance rate ($log_{2}$)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=1)


    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.png')
    plt.tight_layout()
    plt.savefig(out_path)

def freq_thresholding(datas, threshold):
    datas = {key: datas[key] for key in datas if datas[key] <= threshold}
    return datas

def count_cover_rate(all_word_counts, word_counts_subset):
    total_counts        = sum(list(all_word_counts.values()))
    total_subset_counts = sum(list(word_counts_subset.values()))
    return (total_subset_counts / total_counts), total_counts, total_subset_counts

if __name__ == '__main__':

    train_text_path    = "./dump/raw/zh_train_sp/text"
    dev_text_path      = "./dump/raw/zh_dev/text"
    test_text_path     = "./dump/raw/zh_test/text"
    dump_path          = "./exp/test/freq_exp"
    entity_path        = "./local/contextual/rarewords/entity.train.sep.txt"

    train_text_datas = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ') if not d[0].startswith("sp")]
    dev_text_datas   = [[d[0], d[1:]] for d in read_file(dev_text_path, sp=' ') if not d[0].startswith("sp")]
    test_text_datas  = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ') if not d[0].startswith("sp")]
    
    entity_datas      = read_file(entity_path, sp=',')
    entity_datas_dict = {}
    for entity_phrase in entity_datas:
        for word in entity_phrase:
            entity_datas_dict[word] = 1
    entity_datas = list(entity_datas_dict.keys())
    print(f'entity_datas: {entity_datas[:10]}')
    print(f'entity_datas: {len(entity_datas)}')

    # train, dev
    text_datas = train_text_datas + dev_text_datas
    counts = get_word_count(text_datas)

    # counts_log  = {key:np.log2(counts[key]) for key in counts}
    # total_count = sum(list(counts.values()))

    entity_counts = {}
    for word in entity_datas:
        if len(word) < 2:
            continue
        entity_counts[word] = counts[word] if word in counts else 0
    count_value = list(entity_counts.values())
    avg_freq = sum(count_value) / len(count_value)
    print(f'avg_freq: {avg_freq}')
    