import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------
# 一些幫助函式或自定義處理
#-----------------------------------------
def effective_of_number_samples(c):
    """可根據你的需求對原始 counts 做轉換，這裡先直接回傳 c^2 作為示範。"""
    return c ** 3

#-----------------------------------------
# 0. 基本設定
#-----------------------------------------
alpha = 1e-6
N = 500  # 想要在熱力圖中比較的詞數

output_dir = "./exp/statistics/context_distribution_shift"
os.makedirs(output_dir, exist_ok=True)

#-----------------------------------------
# 1. 讀取 Source Domain 資料
#-----------------------------------------
source_context_path = "./local/contextual/contexts/context_S95_keywords_train.txt"
source_context_occurrence_path = "./local/contextual/contexts/context_S95_keywords_train_occurrence.txt"

with open(source_context_path, 'r', encoding='utf-8') as f:
    words_a = [line.strip() for line in f if line.strip()]

with open(source_context_occurrence_path, 'r', encoding='utf-8') as f:
    counts_a = [int(line.strip()) for line in f if line.strip()]

# 轉換成 {詞: 數值}，若計數 > 0 才納入
counts_a_dict = {
    wa: effective_of_number_samples(ca)
    for wa, ca in zip(words_a, counts_a)
    if ca > 0
}

# 用以計算平滑後的詞頻
sum_counts_a = sum(counts_a_dict.values())
vocab_a = set(counts_a_dict.keys())
len_vocab_a = len(vocab_a)

# Smoothed freq for source domain
freq_a = {}
for w in vocab_a:
    ca = counts_a_dict.get(w, 0)
    freq_a[w] = (ca + alpha) / (sum_counts_a + alpha * len_vocab_a)

#-----------------------------------------
# 2. 多個 Target Domain 路徑設定
#    （此為範例，你需自行對應實際檔案）
#-----------------------------------------
target_paths = {
    "Fitness": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_fitness_occurrence.txt",
    ),
    "Design": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_design_occurrence.txt",
    ),
    "Traffic": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_traffic_occurrence.txt",
    ),
    "Child": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_child_occurrence.txt",
    ),
    "Education": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_education_occurrence.txt",
    ),
    "Tradition": (
        "./local/contextual/contexts/context_keywords_test.txt",
        "./local/contextual/contexts/context_keywords_test_tradition_occurrence.txt",
    ),
}

#-----------------------------------------
# 3. 讀取多個 Target Domain，計算 IW
#-----------------------------------------
importance_vectors = []  # 收集各 domain 的 IW(前 N 個詞)
domain_list = []         # 記錄 domain 名稱

# 先收集所有 target vocab 以便排序
all_words_overall = set(vocab_a)
target_dicts = {}  # 暫存每個 target domain 的 (freq_b, vocab_b)

for domain_name, (target_context_path, target_context_occurrence_path) in target_paths.items():
    # 讀取該 domain
    with open(target_context_path, 'r', encoding='utf-8') as f:
        words_b = [line.strip() for line in f if line.strip()]

    with open(target_context_occurrence_path, 'r', encoding='utf-8') as f:
        counts_b = [int(line.strip()) for line in f if line.strip()]

    # 轉換成 {詞: 數值}，若計數 > 0 才納入
    counts_b_dict = {
        wb: effective_of_number_samples(cb)
        for wb, cb in zip(words_b, counts_b)
        if cb > 0
    }
    
    sum_counts_b = sum(counts_b_dict.values())
    vocab_b = set(counts_b_dict.keys())
    len_vocab_b = len(vocab_b)
    
    # smoothing 後的 freq_b
    freq_b = {}
    # 注意：要包含 source vocab 一起考慮 => 做 "vocab_a | vocab_b"
    combined_vocab = vocab_a | vocab_b
    for w in combined_vocab:
        cb = counts_b_dict.get(w, 0)
        freq_b[w] = (cb + alpha) / (sum_counts_b + alpha * len_vocab_b)
    
    target_dicts[domain_name] = (freq_b, combined_vocab)
    all_words_overall |= combined_vocab

# 現在 all_words_overall 包含了所有 domain 可能出現的詞彙
# 為了找到 top N，先對全部詞做「重要性」排序。可根據需求選擇計算方式；
# 這裡示範對所有 target domain 的 freq_b[w] 做總和 + freq_a[w]。
agg_freq = {}
for w in all_words_overall:
    fa = freq_a.get(w, alpha / (sum_counts_a + alpha * len_vocab_a))  
    fb_sum = 0
    for domain_name in target_dicts:
        freq_b, _ = target_dicts[domain_name]
        fb_sum += freq_b.get(w, alpha / 1e10)  # 簡單相加
    agg_freq[w] = fa + fb_sum

# 依重要性排序後選前 N 個詞
sorted_all_words = sorted(agg_freq.keys(), key=lambda w: agg_freq[w], reverse=True)
top_words = sorted_all_words[:N]

# 對每個 domain，計算該 domain 對應 top_words 的 IW 向量
for domain_name in target_dicts:
    freq_b, combined_vocab = target_dicts[domain_name]
    iw_vector = []
    for w in top_words:
        fa = freq_a.get(w, alpha / (sum_counts_a + alpha * len_vocab_a))
        fb = freq_b.get(w, alpha / 1e10)  # 避免 zero
        iw = fb / fa
        iw_vector.append(iw)
    importance_vectors.append(iw_vector)
    domain_list.append(domain_name)

# 轉為 2D array: shape = (num_domains, N)
importance_matrix = np.array(importance_vectors)  # (k, N)

#-----------------------------------------
# 4. 使用熱力圖 (Heatmap) 可視化 - 增強版
#-----------------------------------------

# 取 log10 後再畫 heatmap
log_importance = np.log10(importance_matrix)

# ---- 建議的美化設定 ----
sns.set_style("whitegrid")
# 'notebook' 風格適合一般大小的文字，font_scale 可依需求調整
sns.set_context("notebook", font_scale=1.1)

# 若想要改變整體字型家族 (視系統字體而定)
plt.rcParams["font.family"] = "sans-serif"
# 如果需要支援中文顯示，可在此指定你系統具備的字型，例如微軟正黑體
# plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Heiti TC", "Arial"]

# 建立圖表
fig, ax = plt.subplots(figsize=(8, 6))

# 以 center=0 表示在 log10(1) = 0 作為顏色分界
# 也可嘗試 "coolwarm", "Spectral", "viridis" ... 
sns_heatmap = sns.heatmap(
    log_importance, 
    cmap="RdBu_r",
    center=0,
    # 若 N 很大，可以將 xticklabels 設為 False 或僅顯示部分
    xticklabels=top_words if len(top_words) <= 30 else False,
    yticklabels=domain_list,
    cbar_kws={"label": "Weight (Log Scale)"},  # 顯示 colorbar 標籤
    ax=ax
)

# 假設有顯示 x 軸標籤，可以適度旋轉以避免標籤重疊
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

ax.set_title("Context Importance Weight for Different Domains", fontweight='bold')
ax.set_xlabel(f"Context Word (Sorted by Source Frequency)")
ax.set_ylabel("Target Domains")

# 也可以移除上、右邊框線，讓圖表更乾淨
sns.despine(offset=5, trim=True)

plt.tight_layout()

# 輸出檔案
out_png = os.path.join(output_dir, f"heatmap_log_IW_top_{N}.png")
out_pdf = os.path.join(output_dir, f"heatmap_log_IW_top_{N}.pdf")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Heatmap saved: {out_png}")
