from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Claim your free prize now!"

tokens = tokenizer.tokenize(text)

print("Tokens:")
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs:")
print(token_ids)
