import os
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

##############################################################################
# 0. Global Style Settings
##############################################################################
# Seaborn context settings
output_dir = "./exp/statistics/context_confusion"
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.4)  # Increase or decrease font_scale as needed

# (Optional) Set font family to something like Times New Roman
plt.rcParams['font.family'] = 'sans-serif'

# (Optional) Additional rcParams for consistent styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

def read_json(json_path):
    """Utility function to read a JSON file and return its contents."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

##############################################################################
# 1. Load and Build the Confusion Matrix
##############################################################################
CONFUSION_MATRIX_PATH = 'exp/asr_conformer/run_context_adapter_encoder_suffix/debug/valid/confusion_matrix_final.json'
datas = read_json(CONFUSION_MATRIX_PATH)

# Collect all unique contexts in order of first appearance
context_list = []
for key in datas:
    ground_truth, predicted = key.split('_')
    context_list.append(ground_truth)
    context_list.append(predicted)
context_list = list(dict.fromkeys(context_list))

# Build the full (N x N) matrix
n_classes = len(context_list)
confusion_matrix = np.zeros((n_classes, n_classes), dtype=float)
for key, count in datas.items():
    ground_truth, predicted = key.split('_')
    i = context_list.index(ground_truth)
    j = context_list.index(predicted)
    confusion_matrix[i, j] = count

# (Optional) Apply a transform to manage large values
confusion_matrix = np.clip(confusion_matrix, 0, 600)  # Avoid log(0)
confusion_matrix = np.log10(confusion_matrix + 1)

##############################################################################
# 2. Determine the "Most Confusing" Contexts
##############################################################################
row_sums = confusion_matrix.sum(axis=1)
diag_vals = np.diag(confusion_matrix)
off_diag_sums = row_sums - diag_vals

mask = np.ones_like(confusion_matrix, dtype=bool)
np.fill_diagonal(mask, False)  # We won't count the diagonal
diversity_counts = (confusion_matrix[mask].reshape(n_classes, -1) > 0).sum(axis=1)
# diversity_counts = (confusion_matrix.reshape(n_classes, -1) > 0).sum(axis=1)
confusion_score = off_diag_sums * diversity_counts

K = 10  # Show top K confusing contexts
if n_classes < K:
    K = n_classes

sorted_indices = np.argsort(confusion_score)
top_k_indices = sorted_indices[-K:]

top_l_indices = []
for idx in top_k_indices:
    if idx not in top_l_indices:
        top_l_indices.append(idx)
    top_column_indices = np.argsort(confusion_matrix[idx])[::-1]
    for i in top_column_indices[:3]:
        if confusion_matrix[idx, i] > 0 and i not in top_l_indices:
            top_l_indices.append(i)
top_k_indices = np.array(top_l_indices)
K = top_k_indices.size

# Slice the confusion matrix
confusion_matrix_k = np.zeros((K, K))
row_count, col_count = 0, 0
for i in top_k_indices:
    for j in top_k_indices:
        confusion_matrix_k[row_count, col_count] = confusion_matrix[i, j]
        col_count += 1
    row_count += 1
    col_count = 0

top_k_contexts = [context_list[i] for i in top_k_indices]

##############################################################################
# 3. Build a Directed Graph from the Submatrix
##############################################################################
G = nx.DiGraph()

# Add nodes (with confusion for node sizing/colors)
for idx, context_name in enumerate(top_k_contexts):
    node_confusion = np.clip(confusion_score[top_k_indices[idx]], 0, 40)
    G.add_node(idx, label=context_name, confusion=node_confusion)

# (Optional) Only add edges above a certain threshold
edge_threshold = 0.05
n_top = len(top_k_indices)
for i in range(n_top):
    for j in range(n_top):
        weight = confusion_matrix_k[i, j]
        if weight >= edge_threshold:
            G.add_edge(i, j, weight=weight)

##############################################################################
# 4. Compute a Layout
##############################################################################
# Try spring_layout with a larger 'k' (repulsion) to reduce overlap
pos = nx.spring_layout(G, k=2, seed=42)  
# Alternatively:
# pos = nx.kamada_kawai_layout(G)
# pos = nx.random_layout(G, seed=42)

##############################################################################
# 5. Prepare Node and Edge Styling
##############################################################################
all_node_confusions = np.array([G.nodes[n]['confusion'] for n in G.nodes()])
min_conf, max_conf = all_node_confusions.min(), all_node_confusions.max()

def normalize_node(value, minv, maxv):
    return ((value - minv) / (maxv - minv))

def normalize(value, minv, maxv):
    return ((value - minv) / (maxv - minv)) * 0.8 + 0.2

# Node sizes and colors
node_sizes = []
node_colors = []
node_cmap = plt.cm.Blues  # A warm colormap for nodes
for n in G.nodes():
    c = G.nodes[n]['confusion']
    frac = normalize_node(c, min_conf, max_conf)
    print(f"Node {n}: {G.nodes[n]['label']} - Confusion: {c:.2f} - Fraction: {frac:.2f}")
    node_size = 600 + frac * 1000  # Increased size for clarity
    node_color = node_cmap(frac)
    node_sizes.append(node_size)
    node_colors.append(node_color)

# Edge colors and widths
all_weights = [data['weight'] for _, _, data in G.edges(data=True)]
min_w, max_w = (min(all_weights), max(all_weights)) if all_weights else (0, 1)

edge_cmap = plt.cm.Blues
edge_colors = []
edge_widths = []
edge_alphas = []
for (u, v, data) in G.edges(data=True):
    w = data['weight']
    w_frac = normalize(w, min_w, max_w)
    color = edge_cmap(w_frac)
    # You can scale width by w_frac if desired
    width = 1 + 3 * w_frac  # Thicker edges for higher confusion
    edge_alphas.append(w_frac)
    edge_colors.append(color)
    edge_widths.append(width)

##############################################################################
# 6. Draw the Graph (with Curved Edges)
##############################################################################
plt.figure(figsize=(10, 9))  # Larger figure size for clarity

# Draw edges with curvature
edges = nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    width=edge_widths,
    alpha=edge_alphas,
    connectionstyle='arc3, rad=0.15'
)

# Draw nodes
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.9
)

# Draw labels
# nx.draw_networkx_labels(
#     G, pos,
#     labels={n: G.nodes[n]['label'] for n in G.nodes()},
#     font_size=5,      # Increase if you want larger font for labels
#     font_color='black',
#     font_weight='bold'
# )

# Manually draw labels for each node with dynamic font color
for n in G.nodes():
    label = G.nodes[n]['label']
    c = G.nodes[n]['confusion']
    frac = normalize_node(c, min_conf, max_conf)
    
    # If frac is above 0.6, use white text; otherwise black
    # (Adjust the threshold 0.6 as needed)
    font_color = 'white' if frac > 0.6 else 'black'
    
    # Get (x, y) position of this node
    x, y = pos[n]
    
    # Draw text
    plt.text(x, y, label,
             fontsize=8,
             fontweight='bold',
             color=font_color,
             horizontalalignment='center',
             verticalalignment='center')


plt.title("Top Confusing Contexts (GT -> Pred)", fontsize=16)
plt.axis('off')

##############################################################################
# 7. (Optional) Add Colorbars
##############################################################################
# Edge colorbar
# sm = matplotlib.cm.ScalarMappable(
#     cmap=edge_cmap, 
#     norm=matplotlib.colors.Normalize(vmin=min_w, vmax=max_w)
# )
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.1)
# cbar.set_label("Edge Weight (log10 scale)", fontsize=12)

# Node colorbar
sm_nodes = matplotlib.cm.ScalarMappable(
    cmap=node_cmap,
    norm=matplotlib.colors.Normalize(vmin=0, vmax=1)
)
sm_nodes.set_array([])
cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), fraction=0.03, pad=0.02)
cbar_nodes.set_label("Confusion Score", fontsize=12)

plt.tight_layout()
plt.show()

# Save if desired (commented out by default)
output_path = os.path.join(output_dir, "context_confusion_network_top_k_curved_edges.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
output_path = os.path.join(output_dir, "context_confusion_network_top_k_curved_edges.pdf")
plt.savefig(output_path, bbox_inches='tight')
