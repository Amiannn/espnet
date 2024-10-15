import sentencepiece as spm

# Utility functions for file I/O
from pyscripts.utils.fileio import read_file, write_file

spm_model_path = "./data/token_list/bpe_unigram5000suffix/bpe.model"
s = spm.SentencePieceProcessor(model_file=spm_model_path)

entity_path = "./local/contextual/rarewords/esun_earningcall.entity.txt"

entity_datas = [e[0] for e in read_file(entity_path)]

for entity_data in entity_datas:
    tokens = s.encode(entity_data.lower(), out_type=str)
    print(f'{entity_data}: {tokens}')
