import os
import json
import jieba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn for enhanced styling

from jiwer import cer, wer
from tqdm import tqdm

from pyscripts.utils.fileio import read_file, read_json, write_json
from pyscripts.utils.text_aligner import CheatDetector, align_to_index

# ------------------------------------------------------------------------
# Seaborn setup with a lighter palette
# ------------------------------------------------------------------------
sns.set_theme(
    style="whitegrid",
    palette="bright",
    font_scale=1.5,
    rc={
        "axes.unicode_minus": False,
        "figure.figsize": (12, 8)
    }
)

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.dpi"] = 300          
plt.rcParams["axes.labelsize"] = 12       
plt.rcParams["axes.titlesize"] = 14       
plt.rcParams["legend.fontsize"] = 10      
plt.rcParams["xtick.labelsize"] = 10      
plt.rcParams["ytick.labelsize"] = 10      
plt.rcParams["font.family"] = "sans-serif"

# ------------------------------------------------------------------------
# Paths and file names
# ------------------------------------------------------------------------
TEST_REF_PATH  = './dump/raw/test/text'
TRAIN_REF_PATH = './data/S95/text'
DUMP_PATH      = "./exp/statistics/context_long_tailed"

TEST_RAREWORD_LIST  = './local/contextual/contexts/context_keywords_test.txt'
TEST_UTT_BLIST_PATH = './dump/raw/test/uttblist_idx_keywords'

# Example: you can add more hypotheses here
METHODS = {
    "Baseline": (
        '../asr1/exp/asr_train_conformer_raw_en_bpe5000_sp_suffix/'
        'decode_asr_bs3_asr_model_valid.acc.ave_10best/test/text'
    ),
    "CA": (
        './exp/asr_conformer/run_context_adapter_encoder_suffix/'
        'decode_asr_contextual_bs3_asr_model_valid.acc.ave_10best/test/text'
    ),
    "CA + Importance Weighting ($\mathrm{\hat p}_{\mathcal{T}}(c)=\pi$)": (
        './exp/asr_conformer/run_context_adapter_encoder_reweight0.8_ca_suffix/'
        'decode_asr_contextual_bs3_asr_model_valid.acc.ave_10best/test/text'
    ),
    # "CA + Importance Weighting (CA&Out:long-tailed)": (
    #     './exp/asr_conformer/run_context_adapter_encoder_reweight0.8_suffix/decode_asr_contextual_bs3_asr_model_valid.acc.ave_10best/test/text'
    # ),
    "CA + Importance Weighting ($\mathrm{\hat p}_{\mathcal{T}}(c)=\mathrm{p}_{\mathcal{T}}(c)$)": (
        'exp/asr_conformer/run_context_adapter_encoder_iw_suffix/decode_asr_contextual_bs3_asr_model_valid.acc.ave_10best/test/text'
    ),
}

# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------
def smooth_data(data, kernel_size=20):
    """
    Smooth the input 1D data using a simple moving average.
    """
    kernel = np.ones(kernel_size) / kernel_size
    length = len(data)
    # Duplicate data in reverse, then convolve to handle boundary effects.
    extended = data + data[::-1]
    return np.convolve(extended, kernel, mode='same')[:length]

def get_word_frequency(ref_data, bwords, remove_non_existing=False):
    """
    Count the frequency of each word in 'bwords' according to the reference data.
    """
    freq_dict = {word: 0 for word in bwords}
    for line in tqdm(ref_data, desc="Frequency Counting"):
        if len(line) < 2:
            continue
        _, *ref_words = line
        for w in ref_words:
            if w in freq_dict:
                freq_dict[w] += 1

    # Sort by frequency (descending) for consistency, then reconstruct dict
    sorted_pairs = sorted(((count, w) for w, count in freq_dict.items()), 
                          reverse=True)
    freq_dict    = {w: c for c, w in sorted_pairs}

    # Also track words with zero count, if needed
    zero_count_words = [w for w, c in freq_dict.items() if c == 0]
    if remove_non_existing:
        return freq_dict, zero_count_words
    return freq_dict

def compute_context_error(ref_data, hyp_data, blist_idxs, bwords):
    """
    Compute context errors (substitutions) for each target bword.
    Returns a dict: { bword: [list_of_incorrect_hyps] }
    """
    error_dict = {word: [] for word in bwords}
    for i in tqdm(range(len(ref_data)), desc="Context Error"):
        if len(ref_data[i]) < 2:
            continue
        _, *ref_words = ref_data[i]
        _, *hyp_words = hyp_data[i]
        _, blist_idx  = blist_idxs[i]

        # Get the actual bwords that appear in the reference for this utterance
        blist         = [bwords[idx] for idx in blist_idx]

        # Align references & hypotheses at the token level
        chunks = align_to_index(ref_words, hyp_words)
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref_clean = wref.replace('-', '')
            whyps_clean = ''.join(whyps).replace('-', '')
            # If bword is in the reference and it was misrecognized
            if (wref_clean in blist) and (wref_clean != whyps_clean):
                error_dict[wref_clean].append(whyps_clean)

    return error_dict

def compute_sorted_error_rate(error_dict, train_freq_dict, test_freq_dict):
    """
    Compute WER-based context error rate for each bword, sorted by training frequency.
    
    For each bword with test_freq > 0:
      error_rate(bword) = average( WER(bword, each_misrecognition) ) 
                        = sum(wer(bword, hyp_b)) / test_occurrence
    The final returned list is sorted in the same order as train_freq_dict.
    """
    sorted_error_rate = []
    for bword in train_freq_dict:
        train_occurrence = train_freq_dict[bword]
        test_occurrence  = test_freq_dict.get(bword, 0)
        if test_occurrence == 0:
            error_rate = 0
        else:
            # Sum WER on each misrecognition, then average
            error_rate = sum(wer(bword, hyp_b) for hyp_b in error_dict[bword]) / test_occurrence
        sorted_error_rate.append(error_rate)
    return sorted_error_rate

def plot_error_rates(
    output_path,
    train_occurrences,
    error_rates_dict,
    suffix=''
):
    """
    Plot and compare error rates for multiple methods.
    `error_rates_dict` should be a dict of:
        { "MethodName": [list_of_error_rates], ... }
    all sorted in the same order (matching train_occurrences order).
    """

    # Shots dictionary: threshold => (max_frequency, color_for_shading)
    line_palette = sns.color_palette()
    shots_dict = {
        'many_shot':   [100, line_palette[2]],
        'medium_shot': [20,  line_palette[0]],
        'few_shot':    [1,   line_palette[9]],
        'zero_shot':   [-1,  line_palette[6]],
    }

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Highlight spans (for many/medium/few/zero-shot regions)
    start = 0
    end = 0
    sorted_train_occ = sorted(train_occurrences, reverse=True)  # If needed or not
    # But here we assume `train_occurrences` is *already sorted* to match error_rates
    # If not, handle accordingly.
    for shot_type, (freq_threshold, color) in shots_dict.items():
        for i in range(start, len(train_occurrences)):
            if train_occurrences[i] < freq_threshold:
                break
            end = i
        if shot_type == 'many_shot':
            label_shading = "Many-shot ($n_{c} \u2265 100$)"
        elif shot_type == 'medium_shot':
            label_shading = "Medium-shot ($100 > n_{c} \u2265 20$)"
        elif shot_type == 'few_shot':
            label_shading = "Few-shot ($20 > n_{c} \u2265 1$)"
        else:
            label_shading = "Zero-shot ($n_{c} = 0$)"

        ax.axvspan(xmin=start, xmax=end, facecolor=color, alpha=0.2, label=label_shading)
        start = end

    # Remove duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

    # X-axis setup
    x_vals = np.arange(len(train_occurrences))
    step   = max(1, len(train_occurrences) // 10)
    ax.set_xticks(x_vals[::step])
    ax.set_xticklabels(x_vals[::step], rotation=45)

    # Y-axis setup
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, len(train_occurrences)])
    ax.set_ylabel("Context Error Rate")
    ax.set_xlabel("Sorted Context Index")
    ax.grid(True, linestyle='--', alpha=0.5)

    # We'll pick a palette for the methods
    palette = sns.color_palette("tab10", n_colors=len(error_rates_dict))

    # Smoothing factor
    smoothing_short = 100
    smoothing_long  = 500

    # Plot each method’s error rates
    for (method_name, method_errors), color in zip(error_rates_dict.items(), palette):
        short_smoothed = smooth_data(method_errors, kernel_size=smoothing_short)
        long_smoothed  = smooth_data(method_errors, kernel_size=smoothing_long)

        # Light, behind line
        ax.plot(x_vals, short_smoothed, color=color, linewidth=2, alpha=0.2)
        # White outline for contrast
        ax.plot(x_vals, long_smoothed, color='white', linewidth=3, alpha=1)
        # Actual main line
        ax.plot(x_vals, long_smoothed, color=color, linewidth=2, alpha=0.9, 
                label=method_name)

    ax.set_title("Long-tailed Performance", fontweight='bold', fontsize=12)
    ax.legend(loc="upper left")
    plt.tight_layout()

    # Save figure
    svg_path = os.path.join(output_path, f'word_plot_counts_{suffix}.svg')
    png_path = os.path.join(output_path, f'word_plot_counts_{suffix}.png')
    plt.savefig(svg_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved plots to:\n  {svg_path}\n  {png_path}")

def compute_shot_error_rates(
    error_dict,
    train_freq_dict,
    test_freq_dict,
    many_threshold=100,
    medium_threshold=20
):
    """
    Compute per-shot error rates for a given model's error dictionary.
    For each contextual word w that appears in the test set (test_freq > 0):
      - Use train_freq_dict[w] to determine if it's many/medium/few/zero-shot
      - Sum up the total WER(bword, each_misrecognition)
      - Summation is divided by total occurrences of that word in test
    Finally, each category's rate is the aggregated (weighted) average.

    Returns a dict with keys = ["many","medium","few","zero"] 
    and values = {"rate": float, "sum_wer": float, "sum_occ": int}.
    """
    categories = {
        "many":   {"sum_wer": 0.0, "sum_occ": 0},
        "medium": {"sum_wer": 0.0, "sum_occ": 0},
        "few":    {"sum_wer": 0.0, "sum_occ": 0},
        "zero":   {"sum_wer": 0.0, "sum_occ": 0}
    }

    for bword, t_occ in test_freq_dict.items():
        if t_occ == 0:
            continue
        
        train_occ = train_freq_dict.get(bword, 0)
        if train_occ >= many_threshold:
            shot_label = "many"
        elif train_occ >= medium_threshold:
            shot_label = "medium"
        elif train_occ >= 1:
            shot_label = "few"
        else:
            shot_label = "zero"

        mis_wer = sum(wer(bword, hyp_b) for hyp_b in error_dict[bword])
        
        categories[shot_label]["sum_wer"] += mis_wer
        categories[shot_label]["sum_occ"] += t_occ

    shot_rates = {}
    for cat, vals in categories.items():
        if vals["sum_occ"] > 0:
            rate = vals["sum_wer"] / vals["sum_occ"]
        else:
            rate = 0.0
        shot_rates[cat] = {
            "rate":    rate,
            "sum_wer": vals["sum_wer"],
            "sum_occ": vals["sum_occ"]
        }
    return shot_rates

# ------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------
if __name__ == '__main__':
    # Read reference text
    ref_test_data  = read_file(TEST_REF_PATH, sp=' ')
    ref_train_data = read_file(TRAIN_REF_PATH, sp=' ')

    # Read list of contextual bwords and their indices
    test_blist_idxs = [
        [line[0], [int(x) for x in line[1:] if x != '']] 
        for line in read_file(TEST_UTT_BLIST_PATH, sp=' ')
    ]
    test_bwords = [
        line[0] 
        for line in read_file(TEST_RAREWORD_LIST, sp=',')
    ]

    # Get frequencies
    test_freq_dict, zero_count_bwords = get_word_frequency(
        ref_test_data, test_bwords, remove_non_existing=True
    )
    train_freq_dict = get_word_frequency(ref_train_data, test_bwords)

    # Filter out zero-occurrence words in test for consistent plotting
    test_freq_dict_filtered  = {
        w: test_freq_dict[w] 
        for w in test_freq_dict 
        if w not in zero_count_bwords
    }
    train_freq_dict_filtered = {
        w: train_freq_dict[w] 
        for w in train_freq_dict 
        if w not in zero_count_bwords
    }

    # ------------------------------------------------------------
    # Compute error dictionaries for each method
    # ------------------------------------------------------------
    error_dicts = {}
    for method_name, hyp_path in METHODS.items():
        print(f"\n=== Reading hypotheses for method: {method_name} ===")
        hyp_data = read_file(hyp_path, sp=' ')
        error_dict = compute_context_error(ref_test_data, hyp_data, test_blist_idxs, test_bwords)
        error_dicts[method_name] = error_dict

    # ------------------------------------------------------------
    # Compute sorted error rates (for plotting) for each method
    # ------------------------------------------------------------
    error_rates_dict = {}
    for method_name, edict in error_dicts.items():
        rates = compute_sorted_error_rate(edict, train_freq_dict_filtered, test_freq_dict_filtered)
        error_rates_dict[method_name] = rates

    # Prepare train occurrences in the same sorted order as train_freq_dict_filtered
    # (train_freq_dict_filtered is already sorted by frequency descending).
    occurrences = np.array(list(train_freq_dict_filtered.values()))

    # ------------------------------------------------------------
    # Plot the results (all methods together)
    # ------------------------------------------------------------
    plot_error_rates(
        DUMP_PATH,
        occurrences,
        error_rates_dict,
        suffix="slidespeech"
    )

    # ------------------------------------------------------------
    # Compute & print per-shot error rates for each method
    # ------------------------------------------------------------
    shot_rate_results = {}
    for method_name, edict in error_dicts.items():
        shot_rate_results[method_name] = compute_shot_error_rates(
            edict, train_freq_dict, test_freq_dict
        )

    # Print out results
    for cat in ["many", "medium", "few", "zero"]:
        print(f"=== {cat.upper()}-SHOT RESULTS ===")
        for method_name in METHODS:
            sr = shot_rate_results[method_name][cat]
            print(f"{method_name:10s} {cat}:\n"
                  f"  Rate     = {sr['rate'] * 100:.2f}\n"
                  f"  Sum_WER  = {sr['sum_wer']:.2f}\n"
                  f"  Sum_Occ  = {sr['sum_occ']}")
        print("")
