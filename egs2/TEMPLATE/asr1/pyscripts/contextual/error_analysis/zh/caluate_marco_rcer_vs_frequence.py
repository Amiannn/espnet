import os
import json
import jieba
import numpy as np
import matplotlib.pyplot as plt

from jiwer import cer
from tqdm  import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import write_json

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

exp_freq = f"{2 ** 12}"
# exp_freq = "0"

rareword_list  = './local/contextual/rarewords/rareword_f10_test.txt'
utt_blist_path = './dump/raw/test/uttblist_idx'
ref_path       = './dump/raw/test/text'
hyp_path       = f'/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__mediumbatch_f{exp_freq}_reweight/decode_asr_conformer_adapter_bs5_asr_model_valid.acc.ave_10best/test/text'
# hyp_path       = '/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/decode_asr_conformer_asr_model_valid.acc.ave_10best/test/text'

train_ref_path = './data/train/text'
dump_path      = "./exp/test/freq_exp"

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

def plot_error_rate(output_path, data, data2, tag=''):
    shots_dict = {
        'many_shot'  : [100, 'tab:green'],
        'medium_shot': [20,  'tab:blue'],
        'frew_shot'  : [1,   'tab:cyan'],
        'zero_shot'  : [-1,  'tab:pink'],
    }

    fig, ax1 = plt.subplots(figsize=(15, 8))             
    
    start = 0
    end   = 0
    for shot in shots_dict:
        for i in range(start, len(data)):
            if data[i].item() < shots_dict[shot][0]:
                break
            end = i
        ax1.axvspan(xmin=start, xmax=end, facecolor=shots_dict[shot][1], alpha=0.2)
        start = end

    x_label = range(0, len(data) + 1, 1000)
    color   = 'tab:blue'
    index = list(range(len(data)))
    line1 = ax1.plot(index, data, color=color, linewidth=5, label="Train/Test Context-word Overlap")
    # ax.set_title(f'Word Occurrence - {tag}')
    ax1.set_xlabel('Sorted Context-word Index')
    ax1.set_ylabel('Occurrence', color=color)
    ax1.set_xticks(ticks=x_label, labels=x_label, rotation=45)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.grid(color='gray', linestyle = '--', linewidth=1)
    ax1.legend(loc=2)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0, xmax=len(data))

    ax2   = ax1.twinx() 
    color = 'tab:orange'
    line2 = ax2.plot(index, data2, color=color, linewidth=3, alpha=0.5, label="Context-word Error Rate")
    ax2.set_ylabel('Error Rate', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=1)
    ax2.set_ylim(ymin=0, ymax=1.0)

    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.svg')
    plt.tight_layout()
    plt.savefig(out_path)

def get_frequence(ref_datas, bwords):
    freq_dict = {word: 0 for word in bwords}
    for i in tqdm(range(len(ref_datas))):
        ref_text  = ref_datas[i][1]
        ref_words = list(jieba.cut(ref_text))
        
        for word in ref_words:
            if word in freq_dict:
                freq_dict[word] += 1

    freq_dict = sorted([[freq_dict[w], w] for w in freq_dict], reverse=True)
    freq_dict = {d[1]: d[0] for d in freq_dict}
    return freq_dict

def get_context_error(ref_datas, hyp_datas, bilst_idxs, bwords):
    error_dict = {word: [] for word in bwords}
    for i in tqdm(range(len(ref_datas))):
        ref_text = ref_datas[i][1]
        hyp_text = hyp_datas[i][1]
        blist    = [bwords[idx] for idx in bilst_idxs[i][1]]

        ref_words = list(jieba.cut(ref_text))
        hyp_words = list(jieba.cut(hyp_text))
        chunks    = align_to_index(ref_words, hyp_words)

        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref  = wref.replace('-', '')
            whyps = ''.join(whyps).replace('-', '')
            if (wref in blist) and (wref != whyps):
                error_dict[wref].append(whyps)
    return error_dict

if __name__ == '__main__':
    train_ref_datas = read_file(train_ref_path, sp=' ')

    ref_datas  = read_file(ref_path, sp=' ')
    hyp_datas  = read_file(hyp_path, sp=' ')
    blist_idxs = [[d[0], [int(x) for x in d[1:] if x != '']] for d in read_file(utt_blist_path, sp=' ')]
    bwords     = list(map(lambda x: x[0], read_file(rareword_list, sp=',')))
    
    train_freq_dict = get_frequence(train_ref_datas, bwords)
    freq_dict       = get_frequence(ref_datas, bwords)

    error_dict = get_context_error(ref_datas, hyp_datas, blist_idxs, bwords)

    sorted_error_rate = []
    for bword in train_freq_dict:
        train_occurrence = train_freq_dict[bword]
        test_occurrence  = freq_dict[bword]
        error_count      = len(error_dict[bword])

        error_rate = error_count / test_occurrence
        sorted_error_rate.append(error_rate)
      
    occurence         = np.array(list(train_freq_dict.values()))
    sorted_error_rate = smoothing(sorted_error_rate, kernel_size=20)
    plot_error_rate(dump_path, occurence, sorted_error_rate, tag=f'aishell-train-test-error-rate_f{exp_freq}')

    occurence         = np.array(list(freq_dict.values()))
    sorted_error_rate = smoothing(sorted_error_rate, kernel_size=20)
    plot_error_rate(dump_path, occurence, sorted_error_rate, tag=f'aishell-test-test-error-rate_f{exp_freq}')

    shots_dict = {
        'many_shot'  : [4096, 100],
        'medium_shot': [100, 20],
        'frew_shot'  : [20, 1], 
        'zero_shot'  : [0, -1],
    }

    shot_bwords = {k: [] for k in shots_dict}
    for bword in train_freq_dict:
        occurrence = train_freq_dict[bword]
        for shot in shots_dict:
            smax, smin = shots_dict[shot]
            if smax >= occurrence and occurrence > smin:
                shot_bwords[shot].append(bword)

    print(f'exp_freq: {exp_freq}')
    for shot in shot_bwords:
        shot_rcer = 0
        for bword in shot_bwords[shot]:
            error_word = error_dict[bword]
            occurrence = freq_dict[bword]
            rcer = [cer(bword, eword) for eword in error_word] if len(error_word) > 0 else [0]
            mean_rcer = sum(rcer) / occurrence
            shot_rcer += mean_rcer
            # print(f'{shot}, {occurrence}, {mean_rcer}, {bword}: {error_word}')
        shot_rcer = shot_rcer / len(shot_bwords[shot])
        print(f'{shot}: {shot_rcer:2f}')

    