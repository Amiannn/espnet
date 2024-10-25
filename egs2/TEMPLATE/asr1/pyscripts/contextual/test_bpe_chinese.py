from transformers import AutoTokenizer
import unicodedata

def map_tokens_to_words(token_ids, tokenizer):
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    text      = tokenizer.decode(token_ids)

    last_text  = ""
    token_text = ""
    mapping    = []
    for idx in range(len(tokens)):
        # Get the token string
        token_str = tokenizer.convert_tokens_to_string(tokens[:idx + 1])
        if text.find(token_str) != -1:
            token_text = token_str[len(last_text):]
            last_text  = token_str
        mapping.append([token_ids[idx], token_text])
    return mapping

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small")
text      = "你好，世界！"  # "Hello, World!" in Chinese
token_ids = tokenizer.encode(text, add_special_tokens=False)


# Initialize variables
outputs = map_tokens_to_words(token_ids, tokenizer)
print(outputs)