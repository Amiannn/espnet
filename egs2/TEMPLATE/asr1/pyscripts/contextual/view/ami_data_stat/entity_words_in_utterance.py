# import json
# import matplotlib.pyplot as plt


# with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/process_idx/data.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)


# with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/ihm_train_sp/uttblist', 'r', encoding='utf-8') as file:
#     utterances = file.readlines()


# utterance_word_counts = []
# for utterance in utterances:
#     count = 0
#     for word in data:
#         if word['word'] in utterance:
#             count += 1
#     utterance_word_counts.append(count)


# word_count_frequency = {}
# for count in utterance_word_counts:
#     if count not in word_count_frequency:
#         word_count_frequency[count] = 0
#     word_count_frequency[count] += 1


# plt.figure(figsize=(10, 6))
# plt.bar(word_count_frequency.keys(), word_count_frequency.values(), color='skyblue')
# plt.xlabel('Word Count')
# plt.ylabel('Number of Utterances')
# plt.title('Number of Utterances vs. Word Count')
# plt.xticks(range(max(word_count_frequency.keys()) + 1))
# plt.show()



# plt.savefig('word_count_bar_chart.png')

# plt.show()
import json
import matplotlib.pyplot as plt

with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/process_idx/data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/dump/raw/ihm_train_sp/uttblist', 'r', encoding='utf-8') as file:
    utterances = file.readlines()

utterance_word_counts = []
for utterance in utterances:
    if not utterance.startswith('sp'):
        count = 0
        for word in data:
            if word['word'] in utterance:
                count += 1
        utterance_word_counts.append(count)

word_count_frequency = {}
for count in utterance_word_counts:
    if count not in word_count_frequency:
        word_count_frequency[count] = 0
    word_count_frequency[count] += 1

plt.figure(figsize=(10, 6))
plt.bar(word_count_frequency.keys(), word_count_frequency.values(), color='skyblue')
plt.xlabel('Word Count')
plt.ylabel('Number of Utterances')
plt.title('Number of Utterances vs. Word Count')
plt.xticks(range(max(word_count_frequency.keys()) + 1))
plt.savefig('word_count_bar_chart.png')
plt.show()

plt.savefig('entitywordinutt.png')

plt.show()