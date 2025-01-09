import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def effective_of_number_samples(c):
    return c

# -------------------------
# 1. 讀檔並處理資料
# -------------------------
source_context_path = "./local/contextual/contexts/context_S95_keywords_train.txt"
source_context_occurrence_path = "./local/contextual/contexts/context_S95_keywords_train_occurrence.txt"
target_context_path = "./local/contextual/contexts/context_keywords_test.txt"
target_context_occurrence_path = "./local/contextual/contexts/context_keywords_test_occurrence.txt"
output_dir = "./exp/statistics/context_distribution_shift"
os.makedirs(output_dir, exist_ok=True)

alpha = 1e-6

with open(source_context_path, 'r', encoding='utf-8') as f:
    words_a = [line.strip() for line in f if line.strip()]

with open(source_context_occurrence_path, 'r', encoding='utf-8') as f:
    counts_a = [int(line.strip()) for line in f if line.strip()]

with open(target_context_path, 'r', encoding='utf-8') as f:
    words_b = [line.strip() for line in f if line.strip()]

with open(target_context_occurrence_path, 'r', encoding='utf-8') as f:
    counts_b = [int(line.strip()) for line in f if line.strip()]

counts_a_dict = dict((wa, effective_of_number_samples(ca)) for wa, ca in zip(words_a, counts_a) if ca > 0)
counts_b_dict = dict((wb, effective_of_number_samples(cb)) for wb, cb in zip(words_b, counts_b) if cb > 0)

sum_counts_a = sum(counts_a_dict.values())
sum_counts_b = sum(counts_b_dict.values())

vocab_a = set(counts_a_dict.keys())
vocab_b = set(counts_b_dict.keys())

len_vocab_a = len(vocab_a)
len_vocab_b = len(vocab_b)

all_words = vocab_a | vocab_b

freq_a = {}
freq_b = {}
for w in all_words:
    ca = counts_a_dict.get(w, 0)
    cb = counts_b_dict.get(w, 0)
    # Smoothed frequencies
    freq_a[w] = (ca + alpha) / (sum_counts_a + alpha * len_vocab_a)
    freq_b[w] = (cb + alpha) / (sum_counts_b + alpha * len_vocab_b)

importance_weights = {}
for w in all_words:
    importance_weights[w] = freq_b[w] / freq_a[w]

# 選擇要可視化的關鍵字數量
N = 500
sorted_all_words = sorted(
    all_words,
    key=lambda w: freq_a[w] + freq_b[w],
    reverse=True
)
top_words = sorted_all_words[:N]

# 對齊 A 與 B 的 smoothed frequency
a_counts_aligned = [freq_a[w] for w in top_words]
b_counts_aligned = [freq_b[w] for w in top_words]
importance_weights_aligned = [importance_weights[w] for w in top_words]


# -------------------------
# 2. 繪圖設定與風格
# -------------------------

# 使用 seaborn 風格
sns.set_theme(style="whitegrid")

# 若需要更精細的 rcParams 設定，可在此調整
plt.rcParams["figure.dpi"] = 300           # 提高輸出解析度
plt.rcParams["axes.labelsize"] = 12        # xy label字體大小
plt.rcParams["axes.titlesize"] = 14        # 標題字體大小
plt.rcParams["legend.fontsize"] = 10       # 圖例字體大小
plt.rcParams["xtick.labelsize"] = 10       # x軸刻度字體
plt.rcParams["ytick.labelsize"] = 10       # y軸刻度字體
plt.rcParams["font.family"] = "sans-serif" # 字型，可改成 serif, Times New Roman 等
# 如果需要 LaTeX 數學字體，可以嘗試：
# plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# 建立子圖
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# x 座標
x = np.arange(len(top_words))

# -------------------------
# (a) 分佈 A
# -------------------------
axes[0].plot(x, a_counts_aligned, color='#1f77b4', label='$\mathrm{P}_{\mathcal{S}}(\mathrm{C}_{s})$', linewidth=1.5)
axes[0].fill_between(x, 0, a_counts_aligned, color='#1f77b4', alpha=0.3)
axes[0].set_ylabel("Freq. Source")
axes[0].set_title("Source Context Distribution".format(N), 
                  fontweight='bold', fontsize=12)

# 可加 legend
axes[0].legend(loc='upper right', frameon=True)

# -------------------------
# (b) 分佈 B
# -------------------------
axes[1].plot(x, b_counts_aligned, color='#2ca02c', label='$\mathrm{P}_{\mathcal{T}}(\mathrm{C}_{s})$', linewidth=1.5)
axes[1].fill_between(x, 0, b_counts_aligned, color='#2ca02c', alpha=0.3)
axes[1].set_ylabel("Freq. Target")
axes[1].set_title("Target Context Distribution".format(N),
                  fontweight='bold', fontsize=12)
axes[1].legend(loc='upper right', frameon=True)

# -------------------------
# (c) 重要性權重 (Log Ratio)
# -------------------------
axes[2].plot(x, importance_weights_aligned, color='#9467bd', 
             label='$\mathrm{P}_{\mathcal{T}}(\mathrm{C}_{s})/\mathrm{P}_{\mathcal{S}}(\mathrm{C}_{s})$', linewidth=1.5)

# 在 log scale 下，填充 (fill_between) 需要確保上下界都大於 0
# baseline (ratio = 1)
axes[2].axhline(1.0, color='gray', linestyle='--', linewidth=1.5)

# 填充大於等於1的區域
axes[2].fill_between(
    x, 1.0, importance_weights_aligned,
    where=np.array(importance_weights_aligned) >= 1,
    color='#9467bd', alpha=0.1
)
# 填充小於1的區域
axes[2].fill_between(
    x, 1.0, importance_weights_aligned,
    where=np.array(importance_weights_aligned) < 1,
    color='#ff7f0e', alpha=0.1
)

axes[2].set_ylabel("Weight (Log Scale)")
axes[2].set_title("Context Importance Weights", 
                  fontweight='bold', fontsize=12)
axes[2].legend(loc='upper right', frameon=True)

# 關鍵：將 y 軸設定為對數刻度 (log scale)，而非對資料做 log
axes[2].set_yscale('log')  

# -------------------------
# 3. 調整佈局與輸出
# -------------------------
axes[2].set_xlabel(f"Context Word Index (Sorted by Source Frequency)")
plt.tight_layout()

output_path = os.path.join(output_dir, f"context_distribution_shift_with_importance_top_{N}_enhanced.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
output_path = os.path.join(output_dir, f"context_distribution_shift_with_importance_top_{N}_enhanced.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {output_path}")
