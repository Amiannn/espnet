import matplotlib.pyplot as plt
from collections import Counter


with open("/share/nas165/litingpai/datasets/ami/new_sorted_name.txt", "r") as file:
    phrases = file.readlines()


phrases = [phrase.strip() for phrase in phrases]


word_counts = [len(phrase.split()) for phrase in phrases]


word_count_freq = Counter(word_counts)


word_count_values = list(word_count_freq.keys())
phrase_count_values = list(word_count_freq.values())


plt.figure(figsize=(10, 6))
plt.bar(word_count_values, phrase_count_values, color='skyblue')
plt.xlabel('Word Count')
plt.ylabel('Number of Phrases')
plt.title('Number of Phrases vs Word Count')
plt.xticks(word_count_values)
plt.tight_layout()
plt.show()

plt.savefig('word_in_phrase.png')

plt.show()