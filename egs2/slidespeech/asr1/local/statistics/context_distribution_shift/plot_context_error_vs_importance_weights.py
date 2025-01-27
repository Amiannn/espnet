import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Paths to the result and context files
result_path = '../asr1/exp/asr_train_conformer_raw_en_bpe5000_sp_suffix/decode_asr_bs3_asr_model_valid.acc.ave_10best/test/score_wer/result.txt'
context_path = './local/contextual/contexts/context_keywords_test.txt'
iw_path = './exp/statistics/context_importance_weights/importance_weights.txt'
iw_uniform_path = './exp/statistics/context_importance_weights/importance_weights.txt'

def read_file(file_path, sp=" "):
    """
    Reads a file and splits each line by the specified separator.

    Args:
        file_path (str): Path to the file.
        sp (str): Separator used to split the lines.

    Returns:
        list of lists: Split lines from the file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def read_result(result_path):
    """
    Reads the result file and extracts reference and hypothesis sentences.

    Args:
        result_path (str): Path to the result file.

    Returns:
        tuple: Two lists containing references and hypotheses respectively.
    """
    with open(result_path, 'r') as f:
        lines = f.readlines()

    refs = []
    hyps = []
    for line in lines:
        if "REF:  " in line:
            ref = line.split("REF:  ")[1].strip()
            ref = [r.upper() for r in ref.split(" ") if r != ""]
            refs.append(ref)
        if "HYP:  " in line:
            hyp = line.split("HYP:  ")[1].strip()
            hyp = [h.upper() for h in hyp.split(" ") if h != ""]
            hyps.append(hyp)
    return refs, hyps

def compute_context_error_rates(refs, hyps, contexts):
    """
    Computes both averaged and non-averaged context error rates for each utterance.

    Args:
        refs (list of lists): Reference sentences split into words.
        hyps (list of lists): Hypothesis sentences split into words.
        contexts (list): List of contextual keywords.

    Returns:
        tuple: Two lists containing averaged and non-averaged context error rates respectively.
    """
    averaged_context_errors = []
    non_averaged_context_errors = []
    for idx, (ref, hyp) in enumerate(zip(refs, hyps)):
        context_count = 0
        context_error = 0
        for i in range(len(ref)):
            if ref[i] in contexts:
                context_count += 1
                if i < len(hyp):
                    if hyp[i] != ref[i]:
                        context_error += 1
                else:
                    # Hypothesis is shorter than reference
                    context_error += 1
        # Calculate averaged error rate
        avg_error_rate = (context_error / context_count) if context_count > 0 else 0
        averaged_context_errors.append(avg_error_rate)
        # Non-averaged: total errors
        non_averaged_context_errors.append(context_error)
    return normalize(averaged_context_errors), normalize(non_averaged_context_errors)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) if np.max(data) != np.min(data) else data

def plot_subplots(iw, iw_uniform, avg_errors, non_avg_errors, save_path=None):
    """
    Creates a 2x2 subplot comparing two importance weights with averaged and non-averaged context error rates.

    Args:
        iw (list): Importance weights from the first method.
        iw_uniform (list): Importance weights from the uniform distribution method.
        avg_errors (list): Averaged context error rates per utterance.
        non_avg_errors (list): Non-averaged context error rates per utterance.
        save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    # Set Seaborn theme for publication
    sns.set_theme(style="ticks", context="talk", palette="colorblind")

    # Normalize importance weights for better visualization
    iw_norm = normalize(np.array(iw))
    iw_uniform_norm = normalize(np.array(iw_uniform))

    # Prepare the subplot grid
    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Adjusted size for clarity
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))  # Adjusted size for clarity
    # plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Define subplot configurations
    subplot_configs = [
        {'ax': axes, 'x': iw_norm, 'y': non_avg_errors, 'title': 'Importance Weights vs Context ErrorRate', 'xlabel': 'Importance Weights', 'ylabel': 'Context ErrorRate'},
        # {'ax': axes[0, 1], 'x': iw_uniform_norm, 'y': avg_errors, 'title': 'Uniform Importance Weights vs Averaged Context Error Rate', 'xlabel': 'Uniform Importance Weights (Normalized)', 'ylabel': 'Averaged Context Error Rate'},
        # {'ax': axes[1, 0], 'x': iw_norm, 'y': non_avg_errors, 'title': 'Importance Weights vs Non-Averaged Context Error Rate', 'xlabel': 'Importance Weights (Normalized)', 'ylabel': 'Non-Averaged Context Error Rate'},
        # {'ax': axes[1, 1], 'x': iw_uniform_norm, 'y': non_avg_errors, 'title': 'Uniform Importance Weights vs Non-Averaged Context Error Rate', 'xlabel': 'Uniform Importance Weights (Normalized)', 'ylabel': 'Non-Averaged Context Error Rate'},
    ]

    for config in subplot_configs:
        ax = config['ax']
        x = config['x']
        y = config['y']
        title = config['title']
        xlabel = config['xlabel']
        ylabel = config['ylabel']

        # Scatter plot with regression line
        sns.regplot(
            x=x, 
            y=y, 
            ax=ax, 
            scatter_kws={'s': 30, 'alpha': 0.7, 'edgecolor': 'k'},  # Increased marker size and added edge color
            line_kws={'color': 'red', 'linewidth': 2}
        )
        
        # Compute Pearson correlation
        corr, p_val = pearsonr(x, y)
        
        # Annotate with correlation coefficient
        ax.annotate(
            f'Pearson r = {corr:.2f}\nP-value = {p_val:.2e}', 
            xy=(0.05, 0.95), 
            xycoords='axes fraction',
            fontsize=12, 
            ha='left', 
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
        
        # Set labels and title with consistent font sizes
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        
        # Enhance grid lines
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a super title with increased font size
    plt.suptitle('Comparison of Importance Weights and Context Error Rates', fontsize=18, fontweight='bold')
    
    # Tight layout with space for super title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        # Save as high-resolution PNG
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subplots saved to {save_path}")
    else:
        plt.show()

def main():
    # Read references and hypotheses
    refs, hyps = read_result(result_path)

    # Read importance weights
    iw_datas = [float(d[0]) for d in read_file(iw_path, sp=' ')]
    iw_uniform_datas = [float(d[0]) for d in read_file(iw_uniform_path, sp=' ')]

    # Read context keywords
    contexts = [d[0].upper() for d in read_file(context_path)]

    # Compute context error rates
    avg_context_errors, non_avg_context_errors = compute_context_error_rates(refs, hyps, contexts)

    # Ensure all lists have the same length
    min_length = min(len(avg_context_errors), len(non_avg_context_errors), len(iw_datas), len(iw_uniform_datas))
    avg_context_errors = avg_context_errors[:min_length]
    non_avg_context_errors = non_avg_context_errors[:min_length]
    iw_datas = iw_datas[:min_length]
    iw_uniform_datas = iw_uniform_datas[:min_length]

    # Normalize importance weights for correlation
    iw_normalized = (np.array(iw_datas) - np.min(iw_datas)) / (np.max(iw_datas) - np.min(iw_datas)) if np.max(iw_datas) != np.min(iw_datas) else iw_datas
    iw_uniform_normalized = (np.array(iw_uniform_datas) - np.min(iw_uniform_datas)) / (np.max(iw_uniform_datas) - np.min(iw_uniform_datas)) if np.max(iw_uniform_datas) != np.min(iw_uniform_datas) else iw_uniform_datas

    # Compute Pearson correlation coefficients for each pair
    correlations = {}
    p_values = {}
    # 1. iw vs averaged
    corr, p_val = pearsonr(iw_normalized, avg_context_errors)
    correlations['iw_avg'] = corr
    p_values['iw_avg'] = p_val
    # 2. iw_uniform vs averaged
    corr, p_val = pearsonr(iw_uniform_normalized, avg_context_errors)
    correlations['iw_uniform_avg'] = corr
    p_values['iw_uniform_avg'] = p_val
    # 3. iw vs non-averaged
    corr, p_val = pearsonr(iw_normalized, non_avg_context_errors)
    correlations['iw_non_avg'] = corr
    p_values['iw_non_avg'] = p_val
    # 4. iw_uniform vs non-averaged
    corr, p_val = pearsonr(iw_uniform_normalized, non_avg_context_errors)
    correlations['iw_uniform_non_avg'] = corr
    p_values['iw_uniform_non_avg'] = p_val

    # Print correlation coefficients
    print("Pearson Correlation Coefficients:")
    print(f"IW vs Averaged Context Error Rate: r = {correlations['iw_avg']:.4f}, p-value = {p_values['iw_avg']:.4e}")
    print(f"IW Uniform vs Averaged Context Error Rate: r = {correlations['iw_uniform_avg']:.4f}, p-value = {p_values['iw_uniform_avg']:.4e}")
    print(f"IW vs Non-Averaged Context Error Rate: r = {correlations['iw_non_avg']:.4f}, p-value = {p_values['iw_non_avg']:.4e}")
    print(f"IW Uniform vs Non-Averaged Context Error Rate: r = {correlations['iw_uniform_non_avg']:.4f}, p-value = {p_values['iw_uniform_non_avg']:.4e}")

    output_dir = './exp/statistics/context_importance_weights'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'comparison_subplots.png')
    # Plot the subplots
    plot_subplots(
        iw=iw_datas,
        iw_uniform=iw_uniform_datas,
        avg_errors=avg_context_errors,
        non_avg_errors=non_avg_context_errors,
        save_path=output_path  # Change to 'comparison_subplots.pdf' for vector format
    )

if __name__ == "__main__":
    main()
