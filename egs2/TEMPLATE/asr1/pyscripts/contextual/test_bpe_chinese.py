from transformers import AutoTokenizer
import unicodedata

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small")

text = "你好，世界！"  # "Hello, World!" in Chinese
token_ids = tokenizer.encode(text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
reconstructed_text = tokenizer.decode(token_ids)

# Initialize variables
text_pointer = 0
mappings = []

token_history = []
last_text     = ""
for token in tokens:
    # Get the token string
    token_history.append(token)
    token_str = tokenizer.convert_tokens_to_string(token_history)

    if last_text == "":
        token_text = token_str
    else:
        token_text = token_str[len(last_text):]
    last_text  = token_str

    print(f'token: {token}, text: {token_text}')