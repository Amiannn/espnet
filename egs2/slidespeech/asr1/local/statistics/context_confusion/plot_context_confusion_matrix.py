import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()  # Use the same theme as the Flights example

def read_json(json_path):
    """Utility function to read a JSON file and return its contents."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
CONFUSION_MATRIX_PATH = 'exp/asr_conformer/run_context_adapter_encoder_suffix/debug/valid/confusion_matrix_final.json'
datas = read_json(CONFUSION_MATRIX_PATH)

# -----------------------------------------------------------------------------
# 2. Build the Full Confusion Matrix
# -----------------------------------------------------------------------------
# Collect all unique contexts in order of first appearance.
context_list = []
for key in datas:
    ground_truth, predicted = key.split('_')
    context_list.append(ground_truth)
    context_list.append(predicted)

context_list = list(dict.fromkeys(context_list))  # Remove duplicates, preserve order
n_classes = len(context_list)

confusion_matrix = np.zeros((n_classes, n_classes), dtype=float)
for key, count in datas.items():
    ground_truth, predicted = key.split('_')
    i = context_list.index(ground_truth)
    j = context_list.index(predicted)
    confusion_matrix[i, j] = count

# (Optional) Apply any scaling if needed, e.g. softmax or log
# confusion_matrix = torch.softmax(torch.tensor(confusion_matrix), dim=1).numpy()
confusion_matrix = np.clip(confusion_matrix, 0, 1e3)  # Avoid log(0)
confusion_matrix = np.log10(confusion_matrix + 1)

# -----------------------------------------------------------------------------
# 3. Compute Confusion Magnitude and Diversity
# -----------------------------------------------------------------------------
# A) Off-diagonal sum per row = "Magnitude of confusion"
row_sums = confusion_matrix.sum(axis=1)        # total predictions for each ground truth
diag_vals = np.diag(confusion_matrix)          # correct predictions on the diagonal
off_diag_sums = row_sums - diag_vals           # total misclassifications

# B) Diversity = number of distinct classes confused with
#    i.e., number of columns j != i where confusion_matrix[i, j] > 0
mask = np.ones_like(confusion_matrix, dtype=bool)
np.fill_diagonal(mask, False)  # We won't count the diagonal
diversity_counts = (confusion_matrix[mask].reshape(n_classes, -1) > 0).sum(axis=1)

# Combine them into a single score
# (You can define your own combination: e.g. multiply, add, or define a custom function)
# confusion_score = off_diag_sums * diversity_counts
confusion_score = diversity_counts

# -----------------------------------------------------------------------------
# 4. Select Top 50 Confusing Classes
# -----------------------------------------------------------------------------
top_k = 50  # how many most confusing contexts to select
if n_classes < top_k:
    top_k = n_classes  # In case there aren't 50 classes total

# Indices of the classes sorted by the confusion score
sorted_indices = np.argsort(confusion_score)
top_k_indices = sorted_indices[-top_k:]  # The last K are the largest
top_k_indices = top_k_indices[::-1]       # Reverse the order to get the largest first
# top_k_indices = np.sort(top_k_indices)   # Sort these indices for a nicer matrix ordering

# Slice the confusion matrix to keep only the selected rows and columns
confusion_matrix_k = confusion_matrix[top_k_indices][:, top_k_indices]
top_k_contexts = [context_list[i] for i in top_k_indices]

# -----------------------------------------------------------------------------
# 5. Make a Pandas DataFrame for Plotting
# -----------------------------------------------------------------------------
df_cm = pd.DataFrame(confusion_matrix_k, index=top_k_contexts, columns=top_k_contexts)

# -----------------------------------------------------------------------------
# 6. Plot the Heatmap
# -----------------------------------------------------------------------------
f, ax = plt.subplots(figsize=(17, 15))

sns.heatmap(
    df_cm, 
    annot=False,        # Set True if you'd like to see numeric values
    fmt=".2f",          # Numeric formatting for annotations
    linewidths=0,       # Grid line width
    ax=ax
)

ax.set_title("Top 50 Most Confusing Contexts (Magnitude + Diversity)", pad=20)
ax.set_xlabel("Predicted")
ax.set_ylabel("Ground Truth")
plt.tight_layout()
plt.show()

# (Optional) Save the figure
f.savefig("context_confusion_matrix_top_k_diversity.png", dpi=300)
