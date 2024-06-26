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

# exp_freq = "0"

exp_test_freq =  16
# rareword_list  = './local/contextual/rarewords/rareword_f10_test.txt'
rareword_list  = f'./local/contextual/rarewords/rareword_f{exp_test_freq}_train.txt'
# utt_blist_path = './dump/raw/test/uttblist_idx'
utt_blist_path = f'./dump/raw/zh_test/uttblist_idx_f{exp_test_freq}'
ref_path       = './dump/raw/test/text'

out_exp_freq = f"{2 ** 4}"
hyp_path     = f'/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__mediumbatch_f{out_exp_freq}_reweight/decode_asr_conformer_adapter_bs5_asr_model_valid.acc.ave_10best/test/text'

hyp_casr_path      = f'/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__mediumbatch_f1024/decode_asr_conformer_adapter_bs5_asr_model_valid.acc.ave_10best/test/text'
hyp_baseline_path  = '/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/decode_asr_conformer_asr_model_valid.acc.ave_10best/test/text'

train_ref_path = './data/train/text'
dump_path      = "./exp/test/freq"

# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

def plot_error_rate(output_path, data, baseline_data, casr_data, our_data, tag=''):
    shots_dict = {
        'many_shot'  : [100, 'tab:green'],
        'medium_shot': [20,  'tab:blue'],
        'frew_shot'  : [1,   'tab:cyan'],
        'zero_shot'  : [-1,  'tab:pink'],
    }

    fig, ax1 = plt.subplots(figsize=(12, 10))             
    
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
    color   = 'tab:green'
    index = list(range(len(data)))
    line1 = ax1.plot(index, data, color=color, linestyle='--', linewidth=5, alpha=0.5, label="Train/test overlap")
    # ax.set_title(f'Word Occurrence - {tag}')
    ax1.set_xlabel('Sorted word index')
    ax1.set_ylabel('Occurrence', color=color)
    ax1.set_xticks(ticks=x_label, labels=x_label, rotation=45)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.grid(color='gray', linestyle = '--', linewidth=1)
    ax1.legend(loc=2)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0, xmax=len(data))

    ax2   = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('CER', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0, ymax=0.5)

    smoothing_factor = 2000

    baseline_data_smoothed     = smoothing(baseline_data, kernel_size=100)
    baseline_data_smoothed_max = smoothing(baseline_data, kernel_size=smoothing_factor)
    ax2.plot(index, baseline_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax2.plot(index, baseline_data_smoothed_max, color=color, alpha=0.9, linewidth=5, label="Baseline")

    # color = 'tab:cyan'
    # casr_data_smoothed     = smoothing(casr_data, kernel_size=100)
    # casr_data_smoothed_max = smoothing(casr_data, kernel_size=smoothing_factor)
    # ax2.plot(index, casr_data_smoothed, color=color, linewidth=3, alpha=0.1)
    # ax2.plot(index, casr_data_smoothed_max, color=color, linewidth=5, alpha=0.9, label="Contextual adapter")

    color = 'tab:pink'
    our_data_smoothed     = smoothing(our_data, kernel_size=100)
    our_data_smoothed_max = smoothing(our_data, kernel_size=smoothing_factor)
    ax2.plot(index, our_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax2.plot(index, our_data_smoothed_max, color=color, linewidth=5, alpha=0.9, label="Ours")
    ax2.legend(loc=1)
    
    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.svg')
    plt.tight_layout()
    plt.savefig(out_path)

def get_frequence(ref_test_datas, bwords):
    freq_dict = {word: 0 for word in bwords}
    for i in tqdm(range(len(ref_test_datas))):
        ref_text  = ref_test_datas[i][1]
        ref_words = list(jieba.cut(ref_text))
        
        for word in ref_words:
            if word in freq_dict:
                freq_dict[word] += 1

    freq_dict = sorted([[freq_dict[w], w] for w in freq_dict], reverse=True)
    freq_dict = {d[1]: d[0] for d in freq_dict}
    return freq_dict

def get_context_error(ref_test_datas, hyp_datas, bilst_idxs, bwords):
    error_dict = {word: [] for word in bwords}
    for i in tqdm(range(len(ref_test_datas))):
        ref_text = ref_test_datas[i][1]
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

def get_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict):
    sorted_error_rate = []
    for bword in train_freq_dict:
        train_occurrence = train_freq_dict[bword]
        test_occurrence  = test_freq_dict[bword]
        # error_count      = len(error_dict[bword])
        # error_rate = error_count / test_occurrence
        if test_occurrence == 0:
            error_rate = 0
        else:
            error_rate = sum([cer(bword, hyp_b) for hyp_b in error_dict[bword]]) / test_occurrence
        sorted_error_rate.append(error_rate)
    return sorted_error_rate

def get_total_bword(datas):
    total_words = []
    for i in tqdm(range(len(datas))):
        text  = datas[i][1]
        words = list(jieba.cut(text))
        words = [word for word in words if len(word) > 1]
        total_words.extend(words)
    total_words = list(set(total_words))
    return total_words

if __name__ == '__main__':
    ref_test_datas     = read_file(ref_path, sp=' ')
    ref_train_datas    = read_file(train_ref_path, sp=' ')
    hyp_datas          = read_file(hyp_path, sp=' ')
    hyp_casr_datas     = read_file(hyp_casr_path, sp=' ')
    hyp_baseline_datas = read_file(hyp_baseline_path, sp=' ')


    blist_idxs = [[d[0], [int(x) for x in d[1:] if x != '']] for d in read_file(utt_blist_path, sp=' ')]
    bwords     = list(map(lambda x: x[0], read_file(rareword_list, sp=',')))

    train_freq_dict = get_frequence(ref_train_datas, bwords)
    test_freq_dict  = get_frequence(ref_test_datas, bwords)

    # baseline
    error_dict          = get_context_error(ref_test_datas, hyp_baseline_datas, blist_idxs, bwords)
    baseline_error_rate = get_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict)

    # casr
    error_dict      = get_context_error(ref_test_datas, hyp_casr_datas, blist_idxs, bwords)
    casr_error_rate = get_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict)

    # our
    error_dict        = get_context_error(ref_test_datas, hyp_datas, blist_idxs, bwords)
    our_error_rate    = get_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict)
    
    occurence = np.array(list(train_freq_dict.values()))
    plot_error_rate(
        dump_path, 
        occurence, 
        baseline_error_rate, 
        casr_error_rate,
        our_error_rate, 
        tag=f'aishell-train-test-error-rate_f{out_exp_freq}'
    )

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

    print(f'exp_freq: {out_exp_freq}')
    for shot in shot_bwords:
        shot_rcer = 0
        count = 0
        for bword in shot_bwords[shot]:
            error_word = error_dict[bword]
            occurrence = test_freq_dict[bword]
            # print(f'{bword}, {error_word}')
            rcer = [cer(bword, eword) for eword in error_word] if len(error_word) > 0 else [0]
            if occurrence > 0:
                shot_rcer += sum(rcer) / occurrence
            # shot_rcer += sum(rcer)
            # count += occurrence
            # print(occurrence)
            # print(f'{shot}, {occurrence}, {mean_rcer}, {bword}: {error_word}')
        if len(shot_bwords[shot]) > 0:
            shot_rcer = shot_rcer / len(shot_bwords[shot])
        # shot_rcer = shot_rcer / count
        print(f'{shot}: {shot_rcer:2f}, {len(shot_bwords[shot])}')

    