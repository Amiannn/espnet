import os
import json
import jieba
import numpy as np
import matplotlib.pyplot as plt

from jiwer import cer
from jiwer import wer
from tqdm  import tqdm

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import write_json

from pyscripts.utils.text_aligner import CheatDetector
from pyscripts.utils.text_aligner import align_to_index

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

test_ref_path  = './dump/raw/test/text'
train_ref_path = './data/train/text'
dump_path      = "./exp/test/freq"

test_gamma          =  10
test_rareword_list  = f'./local/contextual/rarewords/rareword_f{test_gamma}_test.txt'
test_utt_blist_path = f'./dump/raw/zh_test/uttblist_idx_f{test_gamma}'

train_gamma_baseline   = f"{2 ** 4}"
train_gamma_our        = f"{2 ** 12}"
train_balance_baseline = ""
train_balance_our      = "_reweight_lp0.8"

hyp_baseline_path    = '/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/decode_asr_conformer_asr_model_valid.acc.ave_10best/test/text'
hyp_casr_path        = f'/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__mediumbatch_f{train_gamma_baseline}{train_balance_baseline}/decode_asr_conformer_adapter_bs5_asr_model_valid.acc.ave_10best/test/text'
hyp_our_path         = f'/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1_contextual/exp/asr_conformer/adapter__mediumbatch_f{train_gamma_our}{train_balance_our}/decode_asr_conformer_adapter_bs5_asr_model_valid.acc.ave_10best/test/text'

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

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
        if test_occurrence == 0:
            error_rate = 0
        else:
            error_rate = sum([wer(bword, hyp_b) for hyp_b in error_dict[bword]]) / test_occurrence
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

    x_label = range(0, len(baseline_data) + 1, 1000)
    index   = list(range(len(baseline_data)))
    # color   = 'tab:green'
    # line1   = ax1.plot(
    #     index, 
    #     data, 
    #     color=color, 
    #     linewidth=5, 
    #     alpha=0.8, 
    #     label="Train/test word overlap"
    # )
    # ax1.set_xlabel('Sorted word index')
    # ax1.set_ylabel('Occurrence', color=color)
    # ax1.set_xticks(ticks=x_label, labels=x_label, rotation=45)
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.legend(loc=2)
    # ax1.set_ylim(ymin=0)
    # ax1.set_xlim(xmin=0, xmax=len(data))
    
    color = 'tab:blue'
    # ax2   = ax1.twinx() 
    ax1.set_ylabel('Error rate')
    ax1.set_xlabel('Sorted word index')
    ax1.tick_params(axis='y')
    ax1.set_xticks(ticks=x_label, labels=x_label, rotation=45)
    ax1.set_xlim(xmin=0, xmax=len(baseline_data))
    ax1.set_ylim(ymin=0, ymax=0.8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    smoothing_factor = 2000

    baseline_data_smoothed     = smoothing(baseline_data, kernel_size=100)
    baseline_data_smoothed_max = smoothing(baseline_data, kernel_size=smoothing_factor)
    ax1.plot(index, baseline_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax1.plot(index, baseline_data_smoothed_max, color=color, alpha=0.9, linewidth=5, label="Baseline")

    color = 'tab:cyan'
    casr_data_smoothed     = smoothing(casr_data, kernel_size=100)
    casr_data_smoothed_max = smoothing(casr_data, kernel_size=smoothing_factor)
    ax1.plot(index, casr_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax1.plot(index, casr_data_smoothed_max, color=color, linewidth=5, alpha=0.9, label="+ Contextual adapter")

    color = 'tab:pink'
    our_data_smoothed     = smoothing(our_data, kernel_size=100)
    our_data_smoothed_max = smoothing(our_data, kernel_size=smoothing_factor)
    ax1.plot(index, our_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax1.plot(index, our_data_smoothed_max, color=color, linewidth=5, alpha=0.9, label="+ Context-balanced adapter (ours)")
    ax1.legend(loc=1)
    
    plt.tight_layout()
    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.svg')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'word_plot_counts_{tag}.png')
    plt.savefig(out_path)

def display_result(shot_bwords, error_dict, test_freq_dict, tag):
    print(f'_' * 50)
    print(f'{tag}\n')
    for shot in shot_bwords:
        shot_rcer = 0
        count = 0
        for bword in shot_bwords[shot]:
            error_word = error_dict[bword]
            occurrence = test_freq_dict[bword]
            rcer = [cer(bword, eword) for eword in error_word] if len(error_word) > 0 else [0]
            shot_rcer += sum(rcer) / occurrence
        shot_rcer = shot_rcer / len(shot_bwords[shot])
        print(f'{shot}: {shot_rcer:2f}, {len(shot_bwords[shot])}')

if __name__ == '__main__':
    ref_test_datas     = read_file(test_ref_path, sp=' ')
    ref_train_datas    = read_file(train_ref_path, sp=' ')

    hyp_datas          = read_file(hyp_our_path, sp=' ')
    hyp_casr_datas     = read_file(hyp_casr_path, sp=' ')
    hyp_baseline_datas = read_file(hyp_baseline_path, sp=' ')

    test_blist_idxs = [[d[0], [int(x) for x in d[1:] if x != '']] for d in read_file(test_utt_blist_path, sp=' ')]
    test_bwords     = list(map(lambda x: x[0], read_file(test_rareword_list, sp=',')))

    train_freq_dict = get_frequence(ref_train_datas, test_bwords)
    test_freq_dict  = get_frequence(ref_test_datas, test_bwords)

    # baseline
    error_dict_baseline = get_context_error(ref_test_datas, hyp_baseline_datas, test_blist_idxs, test_bwords)
    baseline_error_rate = get_sorted_error_rate(error_dict_baseline, train_freq_dict, test_freq_dict)
    # casr
    error_dict_casr     = get_context_error(ref_test_datas, hyp_casr_datas, test_blist_idxs, test_bwords)
    casr_error_rate     = get_sorted_error_rate(error_dict_casr, train_freq_dict, test_freq_dict)
    # our
    error_dict_our      = get_context_error(ref_test_datas, hyp_datas, test_blist_idxs, test_bwords)
    our_error_rate      = get_sorted_error_rate(error_dict_our, train_freq_dict, test_freq_dict)
    
    occurence = np.array(list(train_freq_dict.values()))
    plot_error_rate(
        dump_path, 
        occurence, 
        baseline_error_rate, 
        casr_error_rate,
        our_error_rate, 
        tag=f'aishell-train-test-error-rate_f{train_gamma_our}_test_f{test_gamma}'
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

    display_result(shot_bwords, error_dict_baseline, test_freq_dict, tag=f'baseline')
    display_result(shot_bwords, error_dict_casr, test_freq_dict, tag=f'casr_gammma:{train_gamma_baseline}')
    display_result(shot_bwords, error_dict_our, test_freq_dict, tag=f'our_gammma:{train_gamma_our}')

    