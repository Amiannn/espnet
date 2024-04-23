
total_sum = 0

with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/exp/asr_stats_raw_en_bpe5000_sp/valid/speech_shape', 'r') as file:
    for line in file:
        line = line.strip().split()
        if len(line) == 2:
            key = line[0]
            value = int(line[1])
            if not key.startswith("sp"):
                total_sum += value


print("total:", total_sum)
