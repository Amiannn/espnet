import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn for enhanced styling
from matplotlib.ticker import FuncFormatter

# Baseline values for each metric
baselines = {
    'WER': 19.96,
    'B-WER': 24.86,
    'U-WER': 19.60
}

# Set up matplotlib parameters for high quality output
plt.rcParams["figure.dpi"] = 300           # Higher output resolution
plt.rcParams["axes.labelsize"] = 12        # xy label font size
plt.rcParams["axes.titlesize"] = 14        # Title font size
plt.rcParams["legend.fontsize"] = 10       # Legend font size
plt.rcParams["xtick.labelsize"] = 10         # x-axis tick label size
plt.rcParams["ytick.labelsize"] = 10         # y-axis tick label size
plt.rcParams["font.family"] = "sans-serif"   # Font family

# Setup the output directory
output_dir = "exp/statistics/distractors"
os.makedirs(output_dir, exist_ok=True)

# 1. Read the CSV file
df = pd.read_csv("exp/results/final_result_aggregate.csv")

# 2. Break down the decode_dir string into separate columns
# Example decode_dir:
# "ca_enc_iw_bbse_out_suffix/decode_asr_bs20_ctx_keyword50_asr_model_valid.acc.ave_10best/test"
df['model_name'] = df['decode_dir'].str.extract(r"(.*?)_suffix/")

# Extract beam_search: digits following "bs" (e.g., bs20 or bs3)
beam_search_extracted = df['decode_dir'].str.extract(r"bs(\d+)", expand=False)
df['beam_search'] = pd.to_numeric(beam_search_extracted, errors='coerce').fillna(0).astype(int)

# Extract inference_context: text between "ctx_" and "_asr_model"
df['inference_context'] = df['decode_dir'].str.extract(r"ctx_(.*?)_asr_model")

# Extract distractor value (assumed to be the number following "keywords_c")
df['distractor'] = df['decode_dir'].str.extract(r"ctx_keywords_c(.*?)_asr")
df['distractor'] = pd.to_numeric(df['distractor'], errors='coerce')

# Create a combined label for each model (for plotting, if needed)
df['model_label'] = df['model_name'] + " (BS" + df['beam_search'].astype(str) + ", " + df['inference_context'] + ")"

# 3. Filter the DataFrame for the desired criteria
df_filtered = df[
    (df['context_type'] == "keywords_fix") & 
    (df['beam_search'] == 20) & 
    (df['inference_context'].str.contains('keywords_c')) & 
    (df['distractor'] != 2000) &
    (df['system'] == "base") &
    ((df['model_name'] == 'ca') | (df['model_name'].str.contains('_all')))
]

# Define the error rate metrics to plot
Metrics = ['WER', 'B-WER', 'U-WER']
metric_map = {
    'WER': 'WER',
    'B-WER': 'Context ErrorRate',
    'U-WER': 'Non-Context ErrorRate',
}
col_map = {
    'ca': 'Contextualized CT',
    'ca_iw_bbse_all': '  + CIW (BBCE)',
    'ca_iw_naive_all': '  + CIW (Naïve)',
    'ca_iw_explicit_all': '  + CIW (Explicit)',
}

# Map model names to the ones specified in col_map
df_filtered = df_filtered.copy()
df_filtered['model_renamed'] = df_filtered['model_name'].map(col_map)

# Plot performance vs distractor size for different models using the "pastel" color palette
for metric in Metrics:
    plt.figure(figsize=(6, 4))
    
    # Add baseline horizontal line for the metric
    baseline_val = baselines.get(metric)
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color='black', linestyle='--', label='CT')

    sns.lineplot(
        data=df_filtered, 
        x='distractor', 
        y=metric, 
        hue='model_renamed',  # use renamed model for the legend
        marker='o', 
        sort=True,
        palette=sns.color_palette("pastel")  # use the pastel color palette
    )
    
    plt.xlabel('Distractor Size')
    plt.ylabel(metric_map[metric])
    plt.legend(title="Model", loc="best")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure as PNG and PDF
    png_path = os.path.join(output_dir, f'{metric}_vs_distractor_by_model.png')
    pdf_path = os.path.join(output_dir, f'{metric}_vs_distractor_by_model.pdf')
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    
    print(f"Saved {metric} plot at:\n  PNG: {png_path}\n  PDF: {pdf_path}")
