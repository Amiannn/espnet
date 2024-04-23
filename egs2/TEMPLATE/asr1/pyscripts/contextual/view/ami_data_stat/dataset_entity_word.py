import json

# 讀取 data.json 中的單詞
with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/process_idx/data.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    words_in_data = [entry['word'] for entry in data]

# 讀取 utttblist.txt 中的單詞
with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/ihm_train_sp/uttblist', 'r', encoding='utf-8') as utttblist_file:
    utttblist = utttblist_file.readlines()
    words_in_utttblist = [line.strip().split()[1:] for line in utttblist]

# 統計在 utttblist 中出現的 entity words
entity_word_count = 0
for words in words_in_utttblist:
    for word in words:
        if word in words_in_data:
            entity_word_count += 1

print("在utttblist中出現的entity word數量:", entity_word_count)
