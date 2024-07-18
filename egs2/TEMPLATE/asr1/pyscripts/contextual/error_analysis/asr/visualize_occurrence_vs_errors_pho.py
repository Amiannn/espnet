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

# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

test_ref_path  = './dump/raw/test_clean/text'
train_ref_path = './data/train_clean_100/text'
dump_path      = "./exp/analysis"

hyp_baseline_path = './exp/asr_train_rnnt_conformer_ngpu4_raw_en_bpe5000_sp/decode_asr_greedy_asr_model_valid.loss.ave_5best/test_clean/text'

filter_pho = ['˥˩', '˧˥', '˥˥', '˨˩˦', '˧', '▁']

def phoneme_split(text):
    # phonemes = list(jieba.cut(text))
    phonemes = text.split(' ')
    return phonemes

def smoothing(data, kernel_size=20):
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    data = data + data[::-1]
    return np.convolve(data, kernel, mode='same')[:length]

def get_error(phoneme1, phoneme2):
    if phoneme1 != phoneme2:
        return 1
    return 0

def get_phonemes(datas):
    phoneme_datas = []
    for phonemes in datas:
        for phoneme in phonemes.split(' '):
            if phoneme in filter_pho:
                continue
            phoneme = phoneme.replace("ˈ", '')
            phoneme = phoneme.replace("ˌ", '')
            phoneme = phoneme.replace("ː", '')
            phoneme = phoneme.replace("ʰ", '')
            phoneme_datas.append(phoneme)
    phoneme_datas = list(set(phoneme_datas))
    return phoneme_datas

def get_frequence(ref_test_datas, bphonemes):
    freq_dict = {phoneme: 0 for phoneme in bphonemes}
    for i in tqdm(range(len(ref_test_datas))):
        ref_phonemes  = ref_test_datas[i]
        
        for phoneme in ref_phonemes.split(' '):
            if phoneme in filter_pho:
                continue
            phoneme = phoneme.replace("ˈ", '')
            phoneme = phoneme.replace("ˌ", '')
            phoneme = phoneme.replace("ː", '')
            phoneme = phoneme.replace("ʰ", '')
            if phoneme in freq_dict:
                freq_dict[phoneme] += 1

    freq_dict = sorted([[freq_dict[w], w] for w in freq_dict], reverse=True)
    freq_dict = {d[1]: d[0] for d in freq_dict}
    return freq_dict

def get_context_error(ref_test_datas, hyp_datas, bphonemes):
    error_dict = {phoneme: [] for phoneme in bphonemes}
    for i in tqdm(range(len(ref_test_datas))):
        ref_phonemes = ref_test_datas[i][1:]
        hyp_phonemes = hyp_datas[i][1:]
        chunks    = align_to_index(ref_phonemes, hyp_phonemes)

        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref  = wref.replace('-', '')
            whyps = ''.join(whyps).replace('-', '')
            if (wref in error_dict) and (wref != whyps):
                error_dict[wref].append(whyps)

    return error_dict

def get_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict):
    sorted_error_rate = []
    for bphoneme in train_freq_dict:
        train_occurrence = train_freq_dict[bphoneme]
        test_occurrence  = test_freq_dict[bphoneme]
        if test_occurrence == 0:
            error_rate = 0
        else:
            error_rate = sum([get_error(bphoneme, hyp_b) for hyp_b in error_dict[bphoneme]]) / test_occurrence
        sorted_error_rate.append(error_rate * 100)
    return sorted_error_rate

def get_total_bphoneme(datas):
    total_phonemes = []
    for i in tqdm(range(len(datas))):
        text  = datas[i][1]
        phonemes = phoneme_split(text)
        phonemes = [phoneme for phoneme in phonemes if len(phoneme) > 1]
        total_phonemes.extend(phonemes)
    total_phonemes = list(set(total_phonemes))
    return total_phonemes

def plot_error_rate(output_path, x, data, baseline_data, tag=''):
    shots_dict = {
        'many_shot'  : [100, 'tab:green'],
        'medium_shot': [20,  'tab:blue'],
        'frew_shot'  : [1,   'tab:cyan'],
        'zero_shot'  : [-1,  'tab:pink'],
    }

    fig, ax1 = plt.subplots(figsize=(16, 8))             
    
    start = 0
    end   = 0
    color = None
    for shot in shots_dict:
        for i in range(start, len(data)):
            if data[i].item() < shots_dict[shot][0]:
                break
            end = i
        ax1.axvspan(xmin=start, xmax=end, facecolor=shots_dict[shot][1], alpha=0.2)
        start = end
    ax1.axvspan(xmin=end, xmax=(len(data) + 1), facecolor=shots_dict["many_shot"][1], alpha=0.2)

    x_label = range(0, len(baseline_data), 1)
    index   = list(range(len(baseline_data)))
    
    color = 'tab:blue'
    ax1.set_ylabel('phoneme-level error rate (%)')
    ax1.set_xlabel('Sorted phoneme index')
    ax1.tick_params(axis='y')
    ax1.set_xticks(ticks=x_label, labels=x)
    ax1.set_xlim(xmin=0, xmax=len(baseline_data))
    ax1.set_ylim(ymin=0, ymax=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    smoothing_factor = 2000

    baseline_data_smoothed     = smoothing(baseline_data, kernel_size=100)
    baseline_data_smoothed_max = smoothing(baseline_data, kernel_size=smoothing_factor)
    ax1.plot(index, baseline_data_smoothed, color=color, linewidth=3, alpha=0.1)
    ax1.plot(index, baseline_data_smoothed_max, color=color, alpha=0.9, linewidth=5, label="ASR model")
    ax1.legend(loc=1)
    
    plt.tight_layout()
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}.svg')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}.png')
    plt.savefig(out_path)
    out_path = os.path.join(output_path, f'phoneme_plot_counts_{tag}.pdf')
    plt.savefig(out_path)

def display_result(shot_bphonemes, error_dict, test_freq_dict, tag):
    print(f'_' * 50)
    print(f'{tag}\n')
    for shot in shot_bphonemes:
        shot_rcer = 0
        count = 0
        for bphoneme in shot_bphonemes[shot]:
            error_phoneme = error_dict[bphoneme]
            occurrence = test_freq_dict[bphoneme]
            rcer = [cer(bphoneme, ephoneme) for ephoneme in error_phoneme] if len(error_phoneme) > 0 else [0]
            shot_rcer += sum(rcer) / occurrence
        shot_rcer = shot_rcer / len(shot_bphonemes[shot])
        print(f'{shot}: {shot_rcer:2f}, {len(shot_bphonemes[shot])}')

if __name__ == '__main__':
    train_text_path    = "./exp/analysis/pho_result_train.json"
    test_text_path     = "./exp/analysis/pho_result.json"
    ref_train_datas    = read_json(train_text_path)['pho']
    ref_test_datas     = [d[0] for d in read_json(test_text_path)['pho']]
    hyp_baseline_datas = [d[1] for d in read_json(test_text_path)['pho']]

    test_phonemes      = get_phonemes(ref_test_datas)
    print(f'test_phonemes: {test_phonemes[:10]}')

    train_freq_dict = get_frequence(ref_train_datas, test_phonemes)
    test_freq_dict  = get_frequence(ref_test_datas, test_phonemes)

    # baseline
    error_dict_baseline = get_context_error(ref_test_datas, hyp_baseline_datas, test_phonemes)
    baseline_error_rate = get_sorted_error_rate(error_dict_baseline, train_freq_dict, test_freq_dict)
    occurence = np.array(list(train_freq_dict.values()))
    
    x = list(train_freq_dict.keys())
    print(x)
    print(len(x))
    plot_error_rate(
        dump_path, 
        x,
        occurence, 
        baseline_error_rate, 
        tag=f'occurrence_vs_error_rate'
    )

    shots_dict = {
        'many_shot'  : [99999999, 100],
        'medium_shot': [100, 20],
        'frew_shot'  : [20, 1], 
        'zero_shot'  : [0, -1],
    }

    shot_bphonemes = {k: [] for k in shots_dict}
    for bphoneme in train_freq_dict:
        occurrence = train_freq_dict[bphoneme]
        for shot in shots_dict:
            smax, smin = shots_dict[shot]
            if smax >= occurrence and occurrence > smin:
                shot_bphonemes[shot].append(bphoneme)

    # display_result(shot_bphonemes, error_dict_baseline, test_freq_dict, tag=f'baseline')