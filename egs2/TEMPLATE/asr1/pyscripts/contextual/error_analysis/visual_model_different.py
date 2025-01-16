import argparse
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pyscripts.utils.fileio import read_file

# Configure plot aesthetics
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def get_datas(paths):
    """Read and process data from the provided paths."""
    datas = {}
    for name in paths:
        data = read_file(paths[name], sp='\t')
        data = {d[0]: float(d[3]) for d in data[1:]}
        datas[name] = data
    return datas

def merge_datas(contexts, datas, ibrs, occ_test):
    """Merge Imbalance Rate and Error Rates into a structured table."""
    table = {ctx: [] for ctx in contexts}
    for ctx in table:
        table[ctx].append(ibrs[ctx])
        for name in datas:
            if ctx in datas[name]:
                table[ctx].append(datas[name][ctx])
            else:
                table[ctx].append(0.0)
    
    # Remove entities that do not appear in the test set
    _table = {}
    for ctx in table:
        if occ_test[ctx] != 0:
            _table[ctx] = table[ctx]
    table = _table

    # Add headers directly here, creating a list of lists format
    title = ["Entity"] + ["ImbalanceRate"] + list(datas.keys())
    table = [[ctx] + table[ctx] for ctx in table]
    table.insert(0, title)  # Insert headers as the first row
    return table

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process data and generate heatmaps.")
    parser.add_argument('--path', action='append', help='Data paths in the format label:path', required=True)
    parser.add_argument('--context-path', help='Path to context data file', required=True)
    parser.add_argument('--occ-train-path', help='Path to occurrence in train set data file', required=True)
    parser.add_argument('--occ-test-path', help='Path to occurrence in test set data file', required=True)
    parser.add_argument('--output', help='Output filename for the heatmap image', default='./sorted_combined_heatmaps.png')
    args = parser.parse_args()

    # Build the paths dictionary
    paths = {}
    for path_arg in args.path:
        try:
            label, path = path_arg.split(':', 1)
            paths[label.strip()] = path.strip()
        except ValueError:
            print(f"Invalid format for --path argument: {path_arg}")
            exit(1)

    # Retrieve and process data
    datas = get_datas(paths)
    context_path  = args.context_path
    context_datas = [d[0] for d in read_file(context_path, sp='\t')] 

    occ_train_path = args.occ_train_path
    occ_test_path  = args.occ_test_path

    occ_train_datas = [int(d[0]) for d in read_file(occ_train_path, sp='\t')] 
    occ_test_datas  = [int(d[0]) for d in read_file(occ_test_path, sp='\t')] 

    # Calculate Imbalance Rate (ibrs)
    ibrs = np.log(np.array(occ_train_datas) * 100 + 1)
    ibrs = 100 - (ibrs / np.max(ibrs)) * 100
    ibrs = {ctx: ibr for ctx, ibr in zip(context_datas, ibrs)}

    # Occurrence in test set
    occ_test = {ctx: occ for ctx, occ in zip(context_datas, occ_test_datas)}

    # Merge and convert into DataFrame
    merged_data = merge_datas(context_datas, datas, ibrs, occ_test)
    error_comparison_df = pd.DataFrame(merged_data[1:], columns=merged_data[0])  # Assign headers

    # Set 'Entity' as the index
    heatmap_data = error_comparison_df.set_index('Entity').fillna(0)

    # Sort the data by the first model in descending order
    first_model_name = list(paths.keys())[0]
    sorted_heatmap_data = heatmap_data.sort_values(by=first_model_name, ascending=False)

    # Extract Imbalance Rate and Error Rates
    imbalance_rates = sorted_heatmap_data[['ImbalanceRate']][:1000]
    error_rates = sorted_heatmap_data.drop(columns=['ImbalanceRate'])[:1000]

    # Enhance Imbalance Rate heatmap by duplicating columns
    num_duplicates = 1  # Number of times to duplicate the imbalance rate column
    imbalance_rates_expanded = pd.concat([imbalance_rates]*num_duplicates, axis=1)
    # Rename columns to avoid duplicate column names
    imbalance_rates_expanded.columns = [f'ImbalanceRate' for i in range(num_duplicates)]

    # Determine the number of entities
    num_entities = len(imbalance_rates_expanded)

    # Create a figure with two grids: one for the imbalance heatmap and one for the error rates heatmap
    fig = plt.figure(figsize=(10, 25))
    # Allocate more width to the imbalance heatmap by adjusting width_ratios
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.1, 4], wspace=0)  # Adjust as needed

    # --- Heatmap for Imbalance Rate on the Left ---
    ax0 = plt.subplot(gs[0])

    # Create a heatmap for Imbalance Rate
    sns.heatmap(imbalance_rates_expanded, ax=ax0, cmap="OrRd", cbar_kws={'label': 'Imbalance Rate (%)'},
                linewidths=.5, linecolor='gray', vmin=0, vmax=100, 
                yticklabels=imbalance_rates_expanded.index, cbar=False)

    # Customize the Imbalance Rate heatmap
    ax0.set_xlabel('', fontsize=16)  # Remove x-label for clarity
    ax0.set_title("", fontsize=18)
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    # --- Heatmap for Error Rates on the Right ---
    ax1 = plt.subplot(gs[1])

    # Create a heatmap for Error Rates
    sns.heatmap(error_rates, ax=ax1, annot=False, cmap="YlGnBu", linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Error Rate (%)'}, vmin=error_rates.min().min(), vmax=error_rates.max().max(), yticklabels=False)

    # Customize the Error Rates heatmap
    ax1.set_xlabel("Models", fontsize=16)
    ax1.set_ylabel("", fontsize=16)
    ax1.set_title("Error Rate Comparison across Models by Entity", fontsize=18)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    # Align y-axis limits between the two heatmaps
    ax1.set_ylim(ax0.get_ylim())

    # Add a super title for the entire figure
    plt.suptitle("Imbalance Rate and Recall@10 Rate by Entity", fontsize=20)

    # Adjust layout to accommodate the super title and prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the combined figure
    plt.savefig(args.output)

    # Display the plot
    plt.show()

"""Usage:
python -m pyscripts.contextual.error_analysis.visual_model_different \
  --path "Conformer:../asr1/exp/asr_train_conformer_raw_en_bpe5000_sp_suffix/decode_asr_bs3_asr_model_valid.acc.ave_10best/test/analysis/error_patterns.tsv" \
  --context-path ./local/contextual/contexts/context_keywords_test.txt \
  --occ-train-path ./local/contextual/contexts/context_keywords_test_occurrence_train.txt \
  --occ-test-path ./local/contextual/contexts/context_keywords_test_occurrence.txt \
  --output ./exp/statistics/context_errors/sorted_combined_heatmaps.png
"""