import torch
from transformers import BertTokenizer
from spam_classifier import SpamClassifier
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = SpamClassifier()

model.load_state_dict(
    torch.load("spam_classifier.pth")
)
model.eval()
text = "Congratulations! You won a free iPhone!"
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)
with torch.no_grad():
    logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
print(logits)
probabilities = torch.softmax(logits, dim=1)

prediction = torch.argmax(logits, dim=1)

if prediction.item() == 1:
    label = "SPAM"
else:
    label = "HAM"

confidence = probabilities[0][prediction.item()].item() * 100

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")
