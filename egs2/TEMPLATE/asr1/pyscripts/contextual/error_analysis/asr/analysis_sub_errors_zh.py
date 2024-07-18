import os
import fuzzy
import difflib
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from nltk import pos_tag, word_tokenize, ne_chunk

from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

from matplotlib_venn import venn3

entity_path      = "../asr1_contextual/local/contextual/rarewords/entity.all.sep.txt"
# common_word_path = "./english-common-words.txt"

# Define categories of errors
error_categories = defaultdict(list)

# Helper functions
def process_entity(datas):
    entity_datas = []
    for data in datas:
        entity = [w for w in data.lower().split(' ') if len(w) > 1]
        if len(entity) > 0:
            entity_datas.extend(entity)
    return list(set(entity_datas))

entity_datas = [d[0] for d in read_file(entity_path, sp=',')]
entity_datas = process_entity(entity_datas)
print(entity_datas[:10])
# common_datas = [d[0] for d in read_file(common_word_path, sp=',')]

def is_numeric(word):
    return word.isdigit()

def is_homophone(pho1, pho2):
    return pho1 == pho2

def is_spelling_error(word1, word2):
    return difflib.SequenceMatcher(None, word1, word2).ratio() > 0.8

def is_entity(word):
    # if word in entity_datas and word not in common_datas:
    if word in entity_datas:
        return True
    return False

def is_phonetic(pho1, pho2):
    return difflib.SequenceMatcher(None, pho1, pho2).ratio() > 0.5

# Detect errors function
def detect_errors(ref_word, hyp_word, ref_pho, hyp_pho):
    error_tag = []
    if ref_word == "" or hyp_word == "":
        return ['Other Errors']
    if ref_word != hyp_word:
        if is_entity(ref_word):
            error_tag.append("Entity Errors")
        if is_numeric(ref_word) and is_numeric(hyp_word):
            error_tag.append("Numeric Errors")
        if is_homophone(ref_pho, hyp_pho):
            error_tag.append("Homophone Errors")
        if is_phonetic(ref_pho, hyp_pho):
            error_tag.append("Phonetic Errors")
        if len(error_tag) == 0:
            error_tag.append("Other Errors")
        return error_tag

if __name__ == '__main__':
    pattern_path = './exp/analysis/error_patterns_pho.json'
    dump_path    = './exp/analysis'
    datas        = read_json(pattern_path)
    
    patterns = datas['errors']

    # Process the data
    for ref_word in tqdm(list(patterns.keys())):
        ref_word_pho = patterns[ref_word]['pho']
        errors  = patterns[ref_word]['errors']
        _errors = [] 
        for hyp_word, hyp_word_pho in errors:
            error_tag = detect_errors(ref_word.lower(), hyp_word.lower(), ref_word_pho, hyp_word_pho)
            _errors.append({
                'hyp': hyp_word,
                'pho': hyp_word_pho,
                'tag': error_tag
            })
        patterns[ref_word]['errors'] = _errors
    
    output_path = os.path.join(dump_path, 'sub_analysis.json')
    write_json(output_path, {
        'metadata': datas['metadata'],
        'patterns': patterns
    })

    error_sets = {
        "Entity Errors": [],
        "Numeric Errors": [],
        "Homophone Errors": [],
        "Phonetic Errors": [],
        "Other Errors": [],
    }
    index = 0
    for ref_word in tqdm(list(patterns.keys())):
        errors = patterns[ref_word]['errors']
        for data in errors:
            tags = data['tag']
            for tag in tags:
                error_sets[tag].append(index)
            index += 1
    
    sets = {}
    for key in [
        "Entity Errors",
        "Phonetic Errors",
        "Homophone Errors",
    ]:
        sets[key] = set(error_sets[key])

    plt.figure(figsize=(8, 8))
    venn = venn3(subsets=[set(s) for s in sets.values()], set_labels=list(sets.keys()))
    
    # Set font size for the labels
    for label in venn.set_labels:
        label.set_fontsize(22)  # Set the font size for set labels
    for label in venn.subset_labels:
        if label:  # Check if label exists
            label.set_fontsize(20)  # Set the font size for subset labels
    
    plt.text(
        0.5, 
        -0.1, 
        f"Other Errors: {len(error_sets['Other Errors'])}", 
        ha='center', 
        va='center', 
        fontsize=20, 
        transform=plt.gca().transAxes
    )

    # Show the plot
    plt.show()

    output_path = os.path.join(dump_path, 'sub_analysis_errors.png')
    plt.savefig(output_path)
    output_path = os.path.join(dump_path, 'sub_analysis_errors.svg')
    plt.savefig(output_path)