import sentencepiece as spm

extract_token = ['<blank>', '<sos/eos>']

sp = spm.SentencePieceProcessor(model_file='data/en_token_list/bpe_unigram600/bpe.model')
vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

vocabs = [extract_token[0]] + vocabs + [extract_token[1]]

with open('data/en_token_list/bpe_unigram5000/tokens_.txt', 'w', encoding='utf-8') as fr:
    for vocab in vocabs:
        fr.write(f'{vocab}\n')