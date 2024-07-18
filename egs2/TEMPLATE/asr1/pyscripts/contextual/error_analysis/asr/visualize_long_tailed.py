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

# entity_path      = "../asr1_contextual/local/contextual/metadata/train100.json"
# common_word_path = "./english-common-words.txt"

# def process_entity(datas):
#     types = {'LOC', 'PER', 'ORG'}
#     entity_datas = []
#     for data in datas:
#         if data['entity_group'] not in types:
#             continue
#         entity = [w for w in data['word'].lower().split(' ') if len(w) > 3]
#         entity = " ".join(entity)
#         if len(entity) > 3:
#             entity_datas.append(entity)
#     return list(set(entity_datas))

# entity_datas = read_json(entity_path)
# entity_datas = process_entity(entity_datas)
# common_datas = [d[0] for d in read_file(common_word_path, sp=',')]

# print(entity_datas[:10])

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

# def is_entity(word):
#     if (word in entity_datas) and (word not in common_datas):
#         return True
#     return False

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
    # plt.xticks(range(max(data.keys()) + 1), rotation=90)
    plt.xticks(rotation=90)

    out_path = os.path.join(output_path, f'word_counts_{tag}.png')
    plt.savefig(out_path)

def plot_freq_count(output_path, data, tag=''):
    fig, ax1 = plt.subplots(figsize=(20, 8))             
    x_label = range(0, len(data) + 1, 2000)

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

    # ax1.plot(entity_x, entity_y, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.png')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.svg')
    plt.savefig(out_path)

def freq_thresholding(datas, threshold):
    datas = {key: datas[key] for key in datas if datas[key] <= threshold}
    return datas

def count_cover_rate(all_word_counts, word_counts_subset):
    total_counts        = sum(list(all_word_counts.values()))
    total_subset_counts = sum(list(word_counts_subset.values()))
    # print(f'total_counts: {total_counts}')
    # print(f'total_subset_counts: {total_subset_counts}')
    return (total_subset_counts / total_counts), total_counts, total_subset_counts

train_text_path    = "./dump/raw/zh_train_sp/text"
dev_text_path      = "./dump/raw/zh_dev/text"
test_text_path     = "./data/zh_test/text"
dump_path          = "./exp/analysis"

train_text_datas = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ') if not d[0].startswith("sp")]
dev_text_datas   = [[d[0], d[1:]] for d in read_file(dev_text_path, sp=' ') if not d[0].startswith("sp")]
test_text_datas  = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ') if not d[0].startswith("sp")]

text_datas = train_text_datas
counts = get_word_count(text_datas)
# counts = get_word_count(test_text_datas)

# top 100
counts_log  = {key:np.log2(counts[key]) for key in counts}

# entity_x = []
# entity_y = []
# for i, word in enumerate(counts):
#     count = counts[word]
#     entity_x.append(i)
#     if is_entity(word.lower()):
#         entity_y.append(np.log2(count))
#     else:
#         entity_y.append(0)
# entity_y = smoothing(entity_y, kernel_size=10)

plot_freq_count(dump_path, list(counts_log.values()), f'aishell_word_occurance')
