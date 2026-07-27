from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = "Claim your free prize now!"

inputs = tokenizer(text, return_tensors="pt")

print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

with torch.no_grad():
    outputs = model(**inputs)

print("\nLast Hidden State Shape:")
print(outputs.last_hidden_state.shape)

print("\nCLS Embedding Shape:")
print(outputs.last_hidden_state[:, 0, :].shape)
