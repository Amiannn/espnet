import sentencepiece as spm
import matplotlib.pyplot as plt


sp = spm.SentencePieceProcessor(model_file='/share/nas165/litingpai/espnet_20240304/espnet/egs2/ami/asr1/data/en_token_list/bpe_unigram5000/bpe.model')


word_list = []
with open('/share/nas165/litingpai/datasets/ami/new_sorted_name.txt', 'r', encoding='utf-8') as file:
    for line in file:
        word_list.append(line.strip())


subword_counts = []
for word in word_list:
    subword_count = len(sp.encode_as_pieces(word))
    subword_counts.append(subword_count)


word_phrase_counts = {}
for count in subword_counts:
    if count in word_phrase_counts:
        word_phrase_counts[count] += 1
    else:
        word_phrase_counts[count] = 1


plt.bar(word_phrase_counts.keys(), word_phrase_counts.values(), align='center', alpha=0.75)
plt.xlabel('Number of Subwords')
plt.ylabel('Number of Word Phrases')
plt.title('Distribution of Word Phrases')
plt.xticks(range(max(subword_counts)+1))


plt.ylim(0, max(word_phrase_counts.values()) + 1)

plt.grid(axis='y', alpha=0.75)


plt.savefig('subword_in_phrase.png')

plt.show()




# Save the bar chart as an image file
plt.savefig('subword1_in_phrase.png')

plt.show()