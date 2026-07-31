import torch
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from spam_dataset import SpamDataset
import torch.nn as nn
from spam_classifier import SpamClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("spam.csv")

texts = df["Message"].tolist()

labels = df["spamORham"].map({
    "ham": 0,
    "spam": 1
}).tolist()


# -----------------------------
# 2. Train / Validation / Test split
# -----------------------------

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

# -----------------------------
# 3. Tokenizer
# -----------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# -----------------------------
# 4. Datasets
# -----------------------------

train_dataset = SpamDataset(
    train_texts,
    train_labels,
    tokenizer
)

val_dataset = SpamDataset(
    val_texts,
    val_labels,
    tokenizer
)

test_dataset = SpamDataset(
    test_texts,
    test_labels,
    tokenizer
)


# -----------------------------
# 5. DataLoaders
# -----------------------------

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# -----------------------------
# 6. Model, loss and optimizer
# -----------------------------

model = SpamClassifier()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-5
)


# -----------------------------
# 7. Training + Validation
# -----------------------------

epochs = 3

model.train()

best_val_accuracy = 0


for epoch in range(epochs):

    train_loss = 0

    # Training
    for batch in train_dataloader:

        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        loss = criterion(
            logits,
            batch["label"].to(device)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    # Validation
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_dataloader:

            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )

            loss = criterion(
                logits,
                batch["label"].to(device)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == batch["label"].to(device)
            ).sum().item()

            total += batch["label"].size(0)

            val_loss += loss.item()


    val_accuracy = correct / total
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

    torch.save(
        model.state_dict(),
        "spam_classifier.pth"
    )

    print(
        f"Epoch {epoch + 1}, "
        f"Training Loss: {train_loss / len(train_dataloader):.4f}, "
        f"Validation Loss: {val_loss / len(val_dataloader):.4f}, "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

print(
    f"Epoch {epoch + 1}, "
    f"Training Loss: {train_loss / len(train_dataloader):.4f}, "
    f"Validation Loss: {val_loss / len(val_dataloader):.4f}, "
    f"Validation Accuracy: {val_accuracy:.4f}"
)

if epoch < epochs - 1:
    model.train()

