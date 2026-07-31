import torch
import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from spam_dataset import SpamDataset
from spam_classifier import SpamClassifier


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load dataset
df = pd.read_csv("spam.csv")

texts = df["Message"].tolist()

labels = df["spamORham"].map({
    "ham": 0,
    "spam": 1
}).tolist()


# Same split used during training
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=0.2,
    random_state=42
)


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

test_dataset = SpamDataset(
    test_texts,
    test_labels,
    tokenizer
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


model = SpamClassifier().to(device)

model.load_state_dict(
    torch.load(
        "spam_classifier.pth",
        map_location=device
    )
)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_dataloader:

        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch["label"].numpy()
        )
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

print("Test Accuracy :", accuracy)
print("Precision     :", precision)
print("Recall        :", recall)
print("F1 Score      :", f1)

print("\nConfusion Matrix")
print(confusion_matrix(
    all_labels,
    all_predictions
))

print("\nClassification Report")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Ham", "Spam"]
))