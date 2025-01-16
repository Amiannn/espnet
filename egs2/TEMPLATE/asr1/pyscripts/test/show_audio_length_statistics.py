import os
import numpy as np
import matplotlib.pyplot as plt

# Path to your speech_shape file
FILE_PATH = "exp/asr_stats_raw_en_bpe5000_sp_suffix/train/text_shape"

sample_rate = 16000
durations = []

# 1) Read the speech_shape file
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # skip empty lines
        if not line:
            continue
        
        # Expected format: uid samples
        parts = line.split()
        uid = parts[0]
        samples = int(parts[1])
        
        # Convert samples to seconds
        duration_sec = samples / sample_rate
        durations.append(duration_sec)

# 2) Calculate basic statistics
durations_array = np.array(durations)
mean_duration = np.mean(durations_array)
median_duration = np.median(durations_array)
min_duration = np.min(durations_array)
max_duration = np.max(durations_array)

print(f"Number of text files: {len(durations)}")
print(f"Mean duration: {mean_duration:.2f} s")
print(f"Median duration: {median_duration:.2f} s")
print(f"Min duration: {min_duration:.2f} s")
print(f"Max duration: {max_duration:.2f} s")

# 3) Plot a histogram of durations
plt.hist(durations_array, bins=10, edgecolor='black')
plt.title('Distribution of Text Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
output_path = os.path.join(FILE_PATH.rsplit('/', 1)[0], 'Text_durations.png')
plt.savefig(output_path)
