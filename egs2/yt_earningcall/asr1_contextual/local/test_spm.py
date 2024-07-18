import os
import sentencepiece as spm

sp_model_path = "./data/token_list/bpe_unigram3949suffix/bpe.model"
sp = spm.SentencePieceProcessor(model_file=sp_model_path)

text = "那 反倒 是 tomorrow esun corporate banking 跟 外幣 的 放款 我們 是 希望 雙雙 的 讓 他 速度"
tokens = sp.encode(text, out_type=str)

print(f'tokens: {",".join(tokens)}')