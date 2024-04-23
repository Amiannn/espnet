import os
import json
import sentencepiece as spm
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyscripts.utils.fileio import read_file

def plot_utterance_entity_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Number of Utterances vs. Entity Count - {tag}')
    plt.xlabel('Number of Entity')
    plt.ylabel('Number of Utterances')
    plt.xticks(range(max(data.keys()) + 1))

    out_path = os.path.join(output_path, f'utterance_vs_rareword_{tag}.png')
    plt.savefig(out_path)

def plot_entity_char_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Entity Character Count - {tag}')
    plt.xlabel('Number of Characters')
    plt.ylabel('Number of Entity')
    plt.xticks(range(max(data.keys()) + 1))

    out_path = os.path.join(output_path, f'rareword_char_count_{tag}.png')
    plt.savefig(out_path)

def plot_entity_subword_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Entity Subword Count - {tag}')
    plt.xlabel('Number of Subwords')
    plt.ylabel('Number of Entity')
    plt.xticks(range(max(data.keys()) + 1))

    out_path = os.path.join(output_path, f'rareword_subword_count_{tag}.png')
    plt.savefig(out_path)

def plot_entity_word_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Entity Word Count - {tag}')
    plt.xlabel('Number of Words')
    plt.ylabel('Number of Entity')
    plt.xticks(range(max(data.keys()) + 1))

    out_path = os.path.join(output_path, f'rareword_word_count_{tag}.png')
    plt.savefig(out_path)

if __name__ == '__main__':
    datasets       = ['zh_train_sp', 'zh_dev', 'zh_test']
    rareword_paths = [
        "./local/contextual/rarewords/rareword.train.sep.txt",
        "./local/contextual/rarewords/rareword.all.sep.txt"
    ]
    spm_path    = './data/zh_token_list/bpe_unigram4500suffix/bpe.model'
    sp          = spm.SentencePieceProcessor(model_file=spm_path)
    output_path = "./exp/test"

    for dataset in datasets:
        uttblist_idx_path = f"./dump/raw/{dataset}/uttblist_idx"
        text_path         = f"./dump/raw/{dataset}/text"
        rareword_path     = rareword_paths[1] if 'test' in dataset else rareword_paths[0]
        
        uttblist_idx = {d[0]: d[1:] for d in read_file(uttblist_idx_path, sp=' ')}
        texts        = {d[0]: d[1:] for d in read_file(text_path, sp=' ')}
        rarewords    = [d[0] for d in read_file(rareword_path, sp='\t')]
        
        word_count = {}
        for uid in tqdm(texts):
            if uid.startswith('sp'): 
                continue
            word_length     = len(texts[uid])
            rareword_length = len([len(w) for w in uttblist_idx[uid] if "".join(w) != ''])
            word_count[rareword_length] = word_count[rareword_length] + 1 if rareword_length in word_count else 1
        plot_utterance_entity_count(output_path, word_count, tag=f'libri_{dataset}')

        count_char_table    = {}
        count_subword_table = {}
        count_word_table    = {}
        count_phrase_table  = {}

        for uid in tqdm(texts):
            blist_idx = uttblist_idx[uid]
            if "".join(blist_idx) == "":
                continue
            words = [rarewords[int(idx)] for idx in blist_idx]
            for word in words:
                # character-level counts
                char_length = len(list(word))
                count_char_table[char_length] = count_char_table[char_length] + 1 if char_length in count_char_table else 1
                # subword-level counts
                subword = sp.encode(word)
                print(subword)
                subword_length = len(subword)
                count_subword_table[subword_length] = count_subword_table[subword_length] + 1 if subword_length in count_subword_table else 1
                # word-level counts
                word_length = len(word.split(' '))
                count_word_table[word_length] = count_word_table[word_length] + 1 if word_length in count_word_table else 1
        
        plot_entity_char_count(output_path, count_char_table, tag=f'aishell_{dataset}')
        plot_entity_subword_count(output_path, count_subword_table, tag=f'aishell_{dataset}')
        plot_entity_word_count(output_path, count_word_table, tag=f'aishell_{dataset}')