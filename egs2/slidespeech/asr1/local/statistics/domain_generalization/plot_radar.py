import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn for enhanced styling
from matplotlib.ticker import FuncFormatter

# Set up matplotlib parameters for high quality output
plt.rcParams["figure.dpi"] = 300           # Higher output resolution
plt.rcParams["axes.labelsize"] = 12        # xy label font size
plt.rcParams["axes.titlesize"] = 14        # Title font size
plt.rcParams["legend.fontsize"] = 10       # Legend font size
plt.rcParams["xtick.labelsize"] = 10       # x-axis tick label size
plt.rcParams["ytick.labelsize"] = 10       # y-axis tick label size
plt.rcParams["font.family"] = "sans-serif" # Font family

# Setup the output directory
output_dir = "exp/statistics/domain_generalization"
os.makedirs(output_dir, exist_ok=True)

# 1. Read the CSV file
df = pd.read_csv("exp/results/domain_result_aggregate.csv")

# 2. Break down the decode_dir string into separate columns
# Example decode_dir:
# "ca_enc_iw_bbse_out_suffix/decode_asr_bs20_ctx_keyword50_asr_model_valid.acc.ave_10best/test"

# Extract model_name: text between "ca_" and "_suffix"
df['model_name'] = df['decode_dir'].str.extract(r"(.*?)_suffix/")

# Extract beam_search: digits following "bs" (e.g., bs20 or bs3)
beam_search_extracted = df['decode_dir'].str.extract(r"bs(\d+)", expand=False)
df['beam_search'] = pd.to_numeric(beam_search_extracted, errors='coerce').fillna(0).astype(int)

# Extract inference_context: text between "ctx_" and "_asr_model"
df['inference_context'] = df['decode_dir'].str.extract(r"ctx_(.*?)_asr_model")

# Create a combined label for each model (for plotting)
df['model_label'] = df['model_name'] + " (BS" + df['beam_search'].astype(str) + ", " + df['inference_context'] + ")"

# 3. Filter the DataFrame for the desired criteria
df_filtered = df[
    (df['context_type'] == "keywords_fix") & 
    (df['beam_search'] == 20) & 
    (df['inference_context'] == "keywords") & 
    (df['system'] == "base") &
    ((df['model_name'] == 'ca') | (df['model_name'].str.contains('_all'))) 
]

# 4. Define the desired order for the sessions (domains)
session_order = ['child', 'design', 'education', 'fitness', 'tradition', 'traffic']

Metrics = ['WER', 'B-WER', 'U-WER']

for metric in Metrics:
    # 5. Pivot the DataFrame using pivot_table to handle duplicates.
    #    - Rows (index) are sessions (domains)
    #    - Columns are the unique model_name values (each representing a different decode_dir)
    #    - The cell values are the metric values (aggregated using mean if duplicates exist)
    df_pivot = df_filtered.pivot_table(index='session', columns='model_name', values=metric, aggfunc='mean')

    # Reorder the pivot table based on the desired session order
    df_pivot = df_pivot.reindex(session_order)

    # 6. Compute the relative improvement relative to a baseline model.
    # Here, we choose the first model in the pivot as the baseline.
    baseline_model = df_pivot.columns[0]
    # Compute the baseline values, replacing zeros with NaN to avoid division by zero
    baseline_values = df_pivot[baseline_model].replace(0, np.nan)

    # Calculate relative improvement in percent:
    # Relative Improvement (%) = ((Baseline - Model) / Baseline) * 100
    df_relative = (df_pivot.subtract(baseline_values, axis=0)
                   .div(baseline_values, axis=0)) * -100

    # Fill remaining NaNs if desired
    df_relative = df_relative.fillna(0)

    # Remove the baseline from the plotting DataFrame to avoid a single dot (0% improvement)
    df_relative_plot = df_relative.drop(columns=[baseline_model])
    print(f'metric: {metric}, df_relative_plot: {df_relative_plot}')

    # 7. Prepare the radar chart

    # Calculate the angle for each domain (vertex)
    N = len(session_order)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create the polar (radar) plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Rotate the plot so that the first domain is at the top and draw clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Choose a Seaborn color palette to assign different colors to each model line
    colors = sns.color_palette("pastel", n_colors=len(df_relative_plot.columns))

    # Mapping for nicer legend labels (adjust as needed)
    col_map = {
        'ca': 'Contextualized CT',
        'ca_iw_bbse_all': '  + CIW (BBCE)',
        'ca_iw_naive_all': '  + CIW (Naïve)',
        'ca_iw_explicit_all': '  + CIW (Explicit)',
    }
    for (col, color) in zip(df_relative_plot.columns, colors):
        # Get the relative improvement values for each domain and close the loop
        rel_values = df_relative_plot[col].tolist()
        rel_values += rel_values[:1]
        
        # Plot the line and lightly fill the area under it
        ax.plot(angles, rel_values, color=color, linewidth=2, linestyle='solid', label=col_map.get(col, col))
        ax.fill(angles, rel_values, color=color, alpha=0.1)

    # Set the domain labels on the axes
    session_order_capitalized = [s.capitalize() for s in session_order]
    ax.set_thetagrids(np.degrees(angles[:-1]), session_order_capitalized, fontsize=12, fontweight='bold')

    # Format radial tick labels to include a percentage sign
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}%'))

    # Customize grid and tick parameters for a professional look
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=10)

    # Add a legend outside the chart
    ax.legend(loc='upper right')

    # Optionally, add a title:
    # metric_map = {
    #     'WER': 'WER',
    #     'B-WER': 'Context ErrorRate',
    #     'U-WER': 'Non-Context ErrorRate',
    # }
    # ax.set_title(f"Relative Improvement in {metric_map[metric]} Relative to Baseline Model", 
    #              size=16, y=1.1, fontweight='bold')

    plt.tight_layout()
    output_path_png = os.path.join(output_dir, f"radar_chart_relative_improvement_{metric}.png")
    plt.savefig(output_path_png)
    output_path_pdf = os.path.join(output_dir, f"radar_chart_relative_improvement_{metric}.pdf")
    plt.savefig(output_path_pdf)
    plt.show()
